# S16 — Orchestrator config shape, defaults and routing (PRD)

**Status:** draft, authored 2026-11-10 by the S16 author seat of the `/review-all` remediation program
(`plans/review-all-program-dark_factory-2026-10-10.md`, stream S16, wave 3, size L, risk medium); critic
findings folded 2026-11-11; the sitting's rulings (2026-11-12, residual answers 2026-11-13) folded
2026-11-13 and fold-verified the same day (main `fceb951e7e`). B + H: nine contracts, a two-way
boundary-test sketch, one design invariant, and ω as the
integration gate. G5 fired on every arm: the blast radius is four packages (shared, orchestrator,
fused-memory, dashboard), there are well over eight mechanisms, the seams are load-bearing (configuration
load and hot reload, the LLM route of every dispatch), and S20, S22 and S24 consume this plan's output. Ruling
R15 governs and is applied, not relitigated; R22 item 5 (the invariant) and R23 (weakened keys) also apply.
Twenty-eight leaves, one of them an operator gate (σ) and one a watcher-skill rule the gate needs (χ); none
is held; none is `high`, so no leaf becomes a
dependency of the trigger-chain gate 6698. Every open question is ruled (§Rulings (sitting of 2026-11-12));
no open-question marker remains.

**Code anchors** verified against main `1f240e6da2` (2026-11-10) by the author pass, against `ba17934861`
(2026-11-11) by the fold, against main `ce6673c7e1` (2026-11-13) by the ruling fold (§Freshness at the
ruling fold), and against main `fceb951e7e` (2026-11-13) by the fold-verify, which re-ran every delivered
check and re-read the symbols its fixes rest on (`shared/config_models.py::UsageCapConfig`,
`shared/usage_gate.py::scope_for`, `scripts/legibility/account_pool.py::_pool_config`,
`orchestrator/evals/configs.py`, `skills/escalation-watcher/SKILL.md`, tasks 6754 and 9521). Main moves fast — cite by symbol; re-locate lines at implementation time. The run pinned
`6551b75aba` (2026-10-08). Every count is labelled with the tree it was measured at.

## Goal

Configuration stops being a flat bag of prefix-encoded names with defaults in two places, and model
routing stops being an orchestrator-only mechanism. Concretely:

- **One home per default.** A pydantic `Field` default is the only statement of a default.
  `orchestrator/src/orchestrator/defaults.yaml` keeps only environment-layered values, and a guard test
  fails when a yaml leaf is anything but a `${VAR}` interpolation of a declared field. The 23 disagreeing
  pairs are resolved here, each with its surviving value named (§D2). A design invariant states the rule
  (§Design invariant).
- **Reloadability declared on the field.** A field's hot-reload tier is a marker on the field, and the
  reloadable set is derived from the markers. One generic engine in `shared` diffs and applies a reload
  for both services; fused-memory's ported copy is deleted, and the orchestrator's copy loses its latent
  `KeyError` on an `Optional` submodel.
- **Subsystems are submodels.** The prefix-encoded flat fields of `OrchestratorConfig` move into
  per-subsystem submodels. A consumer reads `config.scheduler.requeue_cap`, not a name it has to
  find among 211 flat siblings. While the moves land, an old setting name keeps loading in both services
  and is reported as deprecated. After the last move is deployed, one pre-authorised session rewrites every
  operated yaml, and then every old name is refused at load and at hot reload with an error that names the
  new path.
- **One role table.** `roles: dict[RoleName, RoleSettings]` replaces six knob-major submodels, the
  role-major `UnblockAutoConfig` knobs and the five flat `watcher_*` role knobs. `RoleName` and `Backend` are
  shared `StrEnum`s, validated at load and at reload, so `backend: Claude` fails at load instead of at
  the first dispatch.
- **One router for every LLM call.** The role → route resolver and the model allowlist live in
  `shared`. Fused-memory's nine model fields and four provider vocabularies become role rows, so its
  LLM calls get a validated route the same way the orchestrator's do. The judge's typo-to-OpenAI arm
  and AgentLoop's unused SDK adapters are deleted.
- **Fused-memory config tells the truth.** Service-path knobs leave `ReconciliationConfig`, consumers
  read the validated schema directly (the `getattr`/`isinstance` re-validations and restated defaults
  go), and two scheduler knobs that each governed a second, undocumented behaviour get their own
  fields.
- **The scoped-cap model has one spelling.** The Fable model that gets its own per-account cap scope is
  declared once, and `scope_for` decodes the CLI alias `fable` to it, so the nightly preparer's Fable caps
  are scoped like the orchestrator's.

What a user observes when this lands:

- `cd /home/leo/src/dark-factory && .venv/bin/python -c "from orchestrator.config import OrchestratorConfig as C; print(C().max_concurrent_tasks, C.model_fields['max_concurrent_tasks'].default)"` prints `24 24`. Today it prints `24 3`.
- While the moves land, `orchestrator check-config` on a project yaml that still says
  `verify_admission_task_slots: 6` or `budgets: {architect: 20}` reports each key as a deprecated spelling
  of its new path, and the orchestrator runs with the value. No born-at-L2 unknown-key escalation is filed.
- After ξ, the same yaml is refused at load, and a hot reload of it returns `reloaded: false`. The error
  names every old key with its new path (`verify_admission_task_slots` → `verify_admission.task_slots`,
  `budgets.architect` → `roles.architect.budget_usd`). By then σ's session has rewritten every operated
  yaml, so nothing in the fleet is refused.
- `mcp__escalation__reload_config` on a yaml that changes `roles.reviewer.model` reports the path in
  `applied`. A yaml with `roles.reviewer.backend: Claude` is refused with a validation error that names
  the field and the allowed values.
- Fused-memory refuses to start on a config whose recon judge names a model outside its
  `model_routing.allowed_models`, and names the role, the model and the allowlist. Today nothing checks.
  While the moves land, a fused-memory yaml that still says `reconciliation.sonnet_model: opus` loads with
  `roles.recon_stage_standard.model == 'opus'`, and its `reload_config` and `get_status` results list the
  deprecation. After ξ it is refused, naming `roles.recon_stage_standard.model`.
- `routing_decision` events keep their shape, and the decision now also carries `backend` and
  `timeout_secs`. The reviewer collapse exists once, in `shared.roles::config_role`.
- `scope_for('fable', UsageCapConfig())` returns `'claude-fable-5-1'`. Today it returns `None`, so a Fable
  cap hit in the nightly preparer marks the whole account capped.

## Background

The run `review-all-dark_factory-20261008` found that the orchestrator's configuration encodes
subsystem membership only in name prefixes, that the role × knob axis has three orientations, that every
default is written twice with 23 disagreements, that routing governs only the orchestrator's share of
the factory's LLM traffic, that the agent backend is a bare string at every layer, and that the
hot-reload engine exists twice. R15 ruled option 1 (field defaults are the one home) and accepted the rest
of the stream as proposed.

### Freshness (re-measured at `1f240e6da2`, 2026-11-10)

`git diff 6551b75aba..main` touched every anchor file: `config.py` +1,070/−144, `scheduler.py`
+1,593/−2,074, `config/schema.py` +520/−179, `memory_service.py` +806/−402, `invoke.py` +263/−88,
`routing.py` and `reload.py` +49/−31 combined. Every symbol was re-read. What changed:

- **`OrchestratorConfig` grew.** 247 model fields: 39 submodel-typed, 208 flat (226: 31/195 at
  `6551b75aba`). (The fold's classifier at `ba17934861` counts 36/211: it counts the three
  `dict`/`list`-of-model fields as flat. A rule choice, not a disagreement.) The flat-name prefix clusters
  are verify 38, max 19, watcher 15, merge 12, orchestrator 9, steward/fused/orphan/terminal/stranded/main/digest
  5 each, and backfill/dashboard/info 4 each. `RELOADABLE_FIELDS` is now a 338-line block, 220 of them
  comment lines, covering 239 dotted paths. 59 of those are flat top-level names listed by hand. The
  justification "FLAT top-level" still appears 4 times.
- **The engine defect is still latent.** `config.py::_iter_leaves` still descends on the runtime value
  (`isinstance(value, BaseModel)`), and `config.py::diff_config` still indexes `fresh_leaves[path]`.
  `OrchestratorConfig` still has 0 `Optional` submodel fields. `fused_memory/config/reload.py` is 465
  lines, its `RELOADABLE_FIELDS` 259 of them, and it still descends by declared annotation.
- **Defaults** (walk of `defaults.yaml` against `OrchestratorConfig.model_fields`, submodels followed).
  226 leaves. 203 restate their field default, `prices` included: its table is equal entry for entry and
  is pinned by `test_default_price_table_matches_defaults_yaml`. 23 disagree; §D2 names them. The
  disagreeing set shifted since the run: `verify_cold_command_timeout_secs` and `timeouts.reviewer`
  joined it, and `budgets.*`, `max_turns.*` and `review.reviewer_briefing` are now listed separately. 174
  bare submodel constructions appear in `orchestrator/tests` + `escalation/tests` (149 at the run). The
  dashboard still restates 24 in `dashboard/src/dashboard/data/orchestrator.py::_ORCHESTRATOR_DEFAULT_MAX_CONCURRENT_TASKS`.
- **Roles.** Six knob-major submodels × 11 roles = 66 role fields. `TimeoutsConfig` also holds
  `startup_grace_secs` and `working_idle_secs`. `UnblockAutoConfig` carries a seventh orientation-major
  copy (model/budget_usd/timeout_seconds/max_turns/effort/backend), and five `watcher_*` flat fields are
  a third. `simple_task_budget_usd`/`simple_task_max_turns` are still deprecated scalars that
  `OrchestratorConfig._migrate_simple_task_scalars` migrates. The reviewer collapse is implemented twice:
  `routing.py::_config_key`, and the inline `role.name.startswith('reviewer')` in `workflow.py`'s
  `TaskWorkflow._invoke` for timeout and backend. `resolve_route` does 9 `getattr`/`hasattr` lookups on
  the knob-major submodels. `OrchestratorConfig._validate_models_in_allowlist` covers `models.*` and
  `unblock_auto.model`, still skips `watcher_model`, and skips any role whose backend `!= 'claude'`.
  `shared.task_metadata.KNOWN_ROLE_NAMES` holds 12 names (including the dispatch identity
  `reviewer_comprehensive`) and `agents/roles.py::ROLES` 9; the reviewer roles are built as `reviewer_<name>`.
- **Backend strings.** There are 13 `str` backend fields (11 `BackendsConfig`, `watcher_backend`,
  `UnblockAutoConfig.backend`). `invoke.py::invoke_agent` still dispatches through an `if/elif` chain
  and raises `ValueError('Unknown backend')` only at dispatch. `shared/invocation_outcome.py` still has
  codex and gemini arms and no pi arm, and `shared/cli_invoke.py` still makes two `backend == 'claude'`
  comparisons.
- **Fused-memory routing.** There are 9 model fields: `LLMConfig.model`;
  `ReconciliationConfig.agent_llm_model`, `judge_llm_model`, `sonnet_model` and `opus_model` (the
  recon stage tiers, chosen by `ReconciliationHarness._select_tier`); `CuratorConfig.model`;
  `PathScopeAdjudicatorConfig.model`; `WriteTriageConfig.judge_model`; `LinkHealConfig.adjudicator_model`.
  The provider axis has 4 vocabularies: `LLMConfig.provider` `Literal['openai','anthropic']`,
  `agent_llm_provider` and `judge_llm_provider` as bare `str` (`'claude_cli'`), and
  `WriteTriageConfig.judge_provider` `Literal['openai','anthropic'] | None`. The live
  `fused-memory/config/config.yaml` pins the write-triage judge to `openai` / `gpt-6.1-sol`.
  `reconciliation/cli_stage_runner.py` still hard-codes `max_budget_usd=5.0`, and fused-memory still
  makes 0 allowlist checks on its own calls.
- **Recon providers.** `Judge._call_llm` still has `claude_cli` / `anthropic` / `else` → OpenAI.
  `AgentLoop._call_llm` still has `anthropic` / `openai` / `ValueError`, and `AgentLoop.run` returns
  early for `claude_cli`. Production configures `claude_cli` for both. The only production
  `AgentLoop(` caller is `reconciliation/verify.py`. `_OpenAIResponseAdapter` is still present, and the
  `anthropic`/`openai` paths are exercised only by tests. The mutation-journaling path with
  `CircuitBreakerError` that the run named is already gone from `reconciliation/agent_loop.py`.
- **ReconciliationConfig.** The class spans 766 lines and 85 fields. The six service-path knobs
  (`procedural_knowledge_near_dup_guard_enabled`, `procedural_knowledge_near_dup_threshold`,
  `procedural_knowledge_topic_guard_clusters`, `procedural_knowledge_topic_cluster_autoseed_enabled`,
  `procedural_knowledge_topic_cluster_autoseed_retired`, `topic_anchored_recall_enabled`) are still in
  it. A seventh field with the same shape now exists: `deterministic_provenance_allowed_agent_prefixes`,
  read by `middleware/done_provenance_authz.py` and `middleware/task_interceptor.py`. The seed function
  `_default_topic_guard_clusters` spans 568 lines.
- **Consumer re-validation.** `_reconciliation_attr` now has three copies
  (`server/near_duplicate_guard.py`, `services/topic_anchor.py`, `middleware/done_provenance_authz.py`;
  two at the run). `MemoryService._record_content_amend` still returns silently on a non-`int`
  threshold. `MemoryService._record_entity_mint` still warns once. `MemoryService.ensure_entity_node`
  still falls back to `_ENTITY_MINT_DEFAULT_LOCK_TIMEOUT_SECONDS = 5.0`. `topic_anchor.py` still
  restates `_DEFAULT_TOPIC_ANCHOR_ENABLED = True`.
- **Double-duty knobs.** `delivered_checks.max_checks_per_tick` is read as the check budget in
  `Scheduler._compute_delivered_check_cache` and as the record-fetch fan-out in
  `Scheduler._phase_backfill_terminal_dep_records`. `max_external_dep_unresolved_cycles` is read in
  `Scheduler._apply_external_dep_policy` and twice in `Scheduler._phase_backfill_dep_status`, whose
  comment still says the dedicated field "is deferred (config.py is out of scope)".
- **Routing cycle.** `config.py` imports `DEFAULT_ALLOWED_MODELS, DEFAULT_LADDER` from
  `orchestrator.routing`, and `routing.py` imports `OrchestratorConfig, RoutingRule` under
  `TYPE_CHECKING` only. `shared.route_rejection` does not exist yet: S25 λ, task 6766, pending.

### Fold measurements (at `ba17934861`, 2026-11-11)

- **Unparsed keys are dropped silently.** `OrchestratorConfig` and `FusedMemoryConfig` are both
  `extra='ignore'`, as is every fused-memory submodel except `memory_metadata` and
  `write_triage_project_calibrations`. Operated yamls carrying knob-major blocks today: dark-factory
  (`timeouts`, `max_turns`, `budgets`, `models`, `effort`, `unblock_auto`), reify (`budgets`, `max_turns`,
  `effort`, `timeouts`, `unblock_auto`), autopilot-video and know-live (`max_turns`).
- **Readers of the moved fields.** A scan over every tracked `.py`/`.sh`/`.yaml`/`.toml`/`.json` file for
  each moved field as an attribute (`.field`), a string key (`'field'`, the `getattr` and `model_copy`
  forms included), a keyword argument (`field=`) or a yaml key gives these file counts per leaf, before
  `config.py`, the new section module and the golden test: ζ1 32, ζ2 34 (28 fields once S17 ζ deletes
  `stranded_blocked_redispatch_enabled`), ζ3 27, ζ4a 43, ζ4b 61, ζ5a 91, ζ5b 23, ζ6a 92, ζ6b 89.
  Single fields exceed any per-leaf file target on their own: `steward_completion_timeout` is set in 61
  test files, `max_review_cycles` in 56, `merge_verify_breadth` in 44. Most test sites are
  `config.<field> = <value>` assignments on hand-built config doubles. The production readers include
  15 single-line `getattr(<config>, '<field>', default)` reads, among them
  `merge_lane/gates.py`'s `getattr(config, 'merge_verify_breadth', None) == 'full'`.
- **`model_copy` accepts anything.** `OrchestratorConfig().model_copy(update={'no_such_knob': 7})`
  succeeds and reads back 7; 87 test `model_copy(update={'<moved flat name>': …})` sites exist.
- **Second spellings outside the root yaml.** Ten per-module `orchestrator.yaml` files in this repo set
  `test_command`/`lint_command`/`type_check_command`/`verify_command_timeout_secs` (orchestrator's also
  `verify_cold_command_timeout_secs`), read through `config.py::_OVERRIDABLE_FIELDS` into
  `ModuleConfig`. The remote-verify spec applies `field_overrides` through
  `verify_spec_shared.py::_validated_field_overrides` against top-level `OrchestratorConfig.model_fields`.
  The two remote-verify host yamls on leo-laptop set `verify_admission_task_slots`,
  `verify_admission_pytest_n`, `verify_cold_preprovision_command`, `merge_verify_max_concurrent_modules`,
  `concurrent_verify` and `verify_use_cgroup_scope`. Only `ORCH_PROJECT_ROOT` among `ORCH_` env vars maps
  to a field.
- **Other restatements.** `dashboard/src/dashboard/data/scheduler.py::_DEFAULT_LOCK_DEPTH = 2` (the
  survivor is 4; the dashboard otherwise reads the scheduler snapshot's published `lock_depth`).
  `scripts/verify_budget_census.py::_MERGE_BUDGET_SOURCES` reads `merge_verify_cold_command_timeout_secs`
  raw from `defaults.yaml`. 48 test files name `defaults.yaml`; 24 of them parse it, 9 through
  `orchestrator/tests/_packaged_defaults.py::load_packaged_defaults`.
- **Fused-memory surfaces.** Fused-memory has no `check-config` and no unknown-key census, and its
  `get_status` result has no config block today. Readers of `LLMConfig.model`/`provider`:
  `backends/graphiti_client.py`, `backends/mem0_client.py`, `routing/classifier.py`, `routing/router.py`,
  `server/main.py`, `server/write_triage_judge.py`, `services/overreach_shadow.py`,
  `arm_harness/arm_config.py`. The shared recon fixture `tests/reconciliation/harness_fixtures.py`,
  imported by 17 test files, sets `agent_llm_provider: 'anthropic'` and model
  `claude-sonnet-4-20250514`.
- **`shared/cli_invoke.py`** names `max_consecutive_zero_output_timeouts` only in
  `is_zero_output_timeout`'s docstring; it reads no config.
- **Other tasks' checks and files.** Four non-terminal tasks carry delivered checks this plan's moves
  would turn red (9229, 3792, 8143, 8452; 4651 once its fields nest), and 46 non-terminal tasks declare
  `defaults.yaml` in `metadata.files` (§Delivered checks on moved paths).

### Freshness at the ruling fold (re-measured at `ce6673c7e1`, 2026-11-13)

`git log ba17934861..ce6673c7e1` over the plan's 493 paths: 383 commits (322 without merges) change 125
of them; no project yaml
outside this repository was edited after 2026-11-05. Every symbol the plan relies on was re-read. What moved:

- **Two more agent backends.** Task 5399 (done) added the `zcode` backend and task 7993 (done) added `mimo`.
  `agents/invoke.py::invoke_agent` now has six `if/elif` arms (claude, codex, gemini, pi, zcode, mimo) and
  still raises `ValueError('Unknown backend')` only at dispatch; `BackendsConfig`'s docstring lists the six.
  C1 gains `ZCODE` and `MIMO`, and γ1's table covers six CLI backends. `shared/cli_invoke.py::ACCOUNT_POOL_BACKENDS`
  is still `{'claude', 'pi'}` (S20's to type), so both new backends sit outside the account pool.
- **Defaults.** Task 7720 (done) added `MergeDeepConfig.chain_reservation_timeout_secs` (7200) and a
  restating `defaults.yaml` leaf. The walk now finds 227 leaves: 204 restate their field default (`prices`
  equal entry for entry once an absent `cached_input_per_1m` is read as `None`) and the same 23 disagree.
  §D2's survivor table is unchanged.
- **Counts.** `OrchestratorConfig`: 247 fields, 36 submodel-typed and 211 flat (the fold's classifier).
  `RELOADABLE_FIELDS`: a 335-line block (220 comment lines; the "338" in §Freshness was at `1f240e6da2`)
  covering 240 dotted paths, 59 of them flat top-level names; "FLAT top-level" still appears 4 times.
  `config.py` 7,170 lines (1,534 comment lines); `defaults.yaml` 1,043 lines. Fused-memory's
  `config/reload.py` 466 lines (its `RELOADABLE_FIELDS` 260); `ReconciliationConfig` 766 lines and 85
  fields, still holding all seven service-path knobs and the four recon model and two provider fields;
  `_default_topic_guard_clusters` 568 lines; three `_reconciliation_attr` copies;
  `_DEFAULT_TOPIC_ANCHOR_ENABLED`, `_ENTITY_MINT_DEFAULT_LOCK_TIMEOUT_SECONDS`, `cli_stage_runner`'s
  `max_budget_usd=5.0` and `_OpenAIResponseAdapter` all still present. `KNOWN_ROLE_NAMES` 12, `ROLES` 9. The
  routing `TYPE_CHECKING` cycle and both dashboard restatements are still there. `shared.route_rejection`
  (6766) and `INVOCATION_COST_FLOOR_USD` (S20 δ) do not exist yet. `shared/cli_invoke.py` still names
  `max_consecutive_zero_output_timeouts` once, in a docstring; `shared/cli_result.py` does not exist yet.
- **Reader rescans** (the fold's per-field scan, re-run): task 5227 moved the info-L0 router's site drivers
  into `orchestrator/tests/_info_l0_router_fixtures.py` and added `test_info_l0_router_boundary.py`, so
  `test_reviewer_note_dispositions.py` leaves ζ1, ζ6a, ζ6b and γ2, and two router tests leave ζ2. Task
  7666 moved the load-regime census tests into `tests/scripts/test_verify_budget_census_regimes.py` (α,
  ζ4a, ζ5a). Tasks 6078, 7720, 3090, 5399 and 7993 added tests naming moved fields
  (`test_verify_relocated_scope_reap.py`, `test_merge_queue_deep_reservation.py`,
  `test_workflow_scope_requests.py`, `test_eval_zcode_bundle.py`, `test_eval_mimo_code_bundles.py`). A
  new script, `fused-memory/scripts/replay_overreach_extraction.py`, and its test read `config.llm.model`
  and `.provider` (ε2b). β1 also declares `orchestrator/src/orchestrator/cli.py`, which prints
  `check-config`'s census buckets and so must print the new deprecated bucket. Each leaf's `Files:` line is
  updated. Two eval task fixtures (`evals/tasks/df_task_18.json`, `evals/tasks_hard_v2/df_task_18.json`) carry
  a `max_review_cycles` key; that is the eval task format `evals/runner.py::build_eval_orch_config` reads
  with `task.get(...)`, not a config spelling, and it does not change.
- **This repository's yamls.** `dark-factory-orchestrator.yaml` still uses 36 old names (the 2026-11-12
  count; only values changed since). `dashboard/orchestrator.yaml` uses 6 project-level old names (`models`,
  `budgets`, `max_turns`, `max_execute_iterations`, `max_verify_attempts`, `max_review_cycles`); its
  module-override keys stay flat (D5). Operator prose naming a moved orchestrator setting: 94 lines in 10
  instruction-bearing files (`OPERATIONS.md` 41, `ARCHITECTURE.md` 14, `SETUP.md` 12,
  `docs/plan-scoring-and-judge.md` 8, `skills/orchestrate/SKILL.md` 6, `skills/escalation-watcher-auto/SKILL.md` 4,
  `CLAUDE.md` 3, `skills/orchestrate/references/project-setup.md` 3, `docs/task-escalation-state-spec.md` 2,
  `docs/config-retune-gate.md` 1); dated records (`docs/ruling-*/`, corpora, PRDs) are not counted.
  Fused-memory: one line in `docs/task-authoring.md` (η), and `fused-memory/config/config-docker.yaml` sets
  `llm.provider` and `llm.model`, which no leaf declared (ε2b now does).
- **Scoped-cap model names** (from the task-8253 investigation, re-read here):
  `shared/config_models.py::UsageCapConfig.scoped_cap_models` defaults to `['claude-fable-5']`, which
  nothing dispatches; `dark-factory-orchestrator.yaml` sets `[claude-fable-5, claude-fable-5-1]`;
  `scripts/sitting/nightly_prepare.py::DEFAULT_MODEL` and `scripts/legibility/config.py`'s
  `census_synthesis` pass the CLI alias `fable`; `orchestrator/routing.py::FABLE_CANDIDATE_MODEL` is
  `'claude-fable-5-1'`; `shared/usage_gate.py::scope_for` matches the string exactly; and the one-shot pool
  (`scripts/legibility/account_pool.py::_pool_config`) inherits the default. Task 5407 (pending, low) asks
  the same question.
- **Task store.** 4867 is done, so S15 β no longer waits on it. 7720 and 8525 are done (7720 leaves the
  parking row). 9493 (the double-SIGTERM fix) is merged, but it takes effect on the host only through
  operator gate 9521 (blocked at level 2 since 2026-11-13T01:29Z), so S24's interval flip (S24 θ) still
  waits on 9521. It no longer waits on S24 ω: the lead exempted θ's one-line `OPERATIONS.md` edit from the
  same-file chain, so the flip lands as soon as 9521 is done. 5399, 7993 and 6615 are done; 8252 is cancelled and
  8253 is rescoped so that it no longer edits `usage_gate.py`, `config_models.py` or `account_pool.py`.
  47 non-terminal tasks declare `defaults.yaml` (46 on 2026-11-11). Every other task the plan names is
  pending as before (4602 deferred). The delivered checks of 9229, 3792, 8143, 8452, 4651, 8373, 3587 and
  8528 are unchanged. 9229's ancestry now also reaches 8528.

Program rulings this PRD implements without relitigating: R15 (all of S16), R22 item 5 (the
invariant), R23 (no weakened key in this stream), R24, R37 (the cross-repository yaml edit, now the
operator gate σ, as the sitting ruled), R51 (invariant landing order, amended by the lead: ι follows S7 ζ 9154 on
the three invariant files, and the six wave-3 minting tasks take INV-18 to INV-23 in landing order), R54 (per-file
order), R55 (worker.py parking).

## Findings disposition

| Key | Sev | Verdict | Disposition | Note |
|---|---|---|---|---|
| `fk-dae2a050b5d3` | medium | confirmed | leaf ζ5b | Grown since the run (208 flat fields at `1f240e6da2`). β0 and β1 move the tier onto the field. ζ1–ζ6b move the prefix clusters into submodels (§D5), κ migrates this repo's yaml and operator prose, σ the other operated yamls, and ξ refuses the old names (§D6). All of them carry the key, and ζ5b is the last ζ leaf in the chain. |
| `fk-5805ae6f97f1` | medium | confirmed (re-observed) | leaf γ2 | `shared.roles` lands in γ0. γ2 replaces the three orientations with one table and implements the reviewer collapse once; δ moves timeout and backend onto the decision. |
| `fk-2a67692ff456` | medium | confirmed | leaf α | 23 disagreements re-measured at `1f240e6da2`, survivors named in §D2. ι promotes the invariant and also carries the key. |
| `fk-dac0a2a67201` | medium | confirmed | leaf ε2b | δ moves the resolver and allowlist to shared. ε2a (recon roles) and ε2b (curator, adjudicators, write-triage judge, memory LLM) route fused-memory's nine fields through it. Eval drivers stay outside (OQ-5, ruled by the lead). |
| `fk-ba569525f6e0` | medium | confirmed (re-observed, grown) | leaf γ1 | `Backend` lands in γ0 with six CLI members (zcode and mimo landed since the draft, tasks 5399 and 7993); γ1 types every orchestrator backend field and keys `invoke_agent` dispatch on it. The `invocation_outcome`/`cli_invoke` sites are S20's to convert (S20 ν, held on `S16/C1`). The per-backend adapter record is not in scope (OQ-6, ruled by the lead). |
| `fk-34d8da6424f4` | medium | confirmed (re-observed) | leaf ε1 | The provider fields become `Backend`. The judge's `else` → OpenAI arm and AgentLoop's SDK adapters are deleted. The judge's anthropic arm and prose verdict path go too (OQ-3, ruled by the lead). Partly fixed before main: the mutation-journaling path with `CircuitBreakerError` is already gone. |
| `fk-709d4d84ee37` | medium | re-observed (prior row) | leaf ε1 | The same two fields as `fk-34d8da6424f4`, seen as an h12 finding; closed by the same leaf. |
| `fk-c520d5374ba5` | low | confirmed | leaf η | Seven service-path knobs now, not six (`deterministic_provenance_allowed_agent_prefixes` joined them). The seed data leaves `schema.py`. |
| `fk-a3dfde6f0664` | low | confirmed | leaf η | Waits on S26: λ 9194 folds the mint lock timeout into `identity_lock`, η 9174 re-signs `resolve_topic_anchor_enabled`, and τ 9245 and υ 9247 move the storm methods. η deletes what remains. |
| `fk-efae65b1f9af` | low | confirmed | leaf β1 | β0 hoists the engine, which fixes the latent `Optional`-submodel `KeyError`. β2 deletes `fused_memory/config/reload.py`; β1, which depends on β2, deletes the orchestrator copy and closes the key. |
| `fk-da71e5bb1679` | low | confirmed | leaf ζ3 | The two dedicated fields land in the scheduler submodel leaf, defaulting to today's shared values. |

No key in this stream is weakened (R23). No finding was fixed on main between the draft and the ruling
fold. No Phase-5 mechanical task carries any of these keys: contract §8 step 1 (`find_tasks_by_metadata`,
`x_finding_key`) returned no task for any of the 11 keys on 2026-11-10, 2026-11-11 and again on
2026-11-13. Step 2 (`search_tasks` ≥ 0.6, one paraphrase per leaf family) returned only done tasks 2005,
2006 and 4771 (history, not duplicates). For the ruling fold's three new leaves, step 2 found task 5407
(pending, low: "`usage_cap.scoped_cap_models` still names the stale `claude-fable-5` literal", score 0.70),
which π absorbs (cancel with a reason naming π), and nothing live for the yaml sweep or the refusal (the
nearest, 2698, 2719 and 7138, are done or are about config file names, not setting names).

## Resolved design decisions

Edge notation throughout this PRD: "X → Y" means X depends on Y (Y lands first).

- **D1 — Order of the work: shared contracts first, then orchestrator, then fused-memory.** β0 (field
  declarations, table pivots and reload engine) and γ0 (role and backend vocabulary) touch only new
  `shared/` modules, so they wait on nothing. Every orchestrator leaf edits `config.py`, so the
  orchestrator leaves form one chain on that file: κ → ζ5b → ζ5a → ζ4b → ζ6b → ζ6a → ζ4a → ζ3 → ζ2 → ζ1 →
  δ → γ2 → γ1 → β1 → α. The three leaves that declare `merge_lane/worker.py` (ζ4b, ζ5a, ζ5b) sit last, so
  the R55 parking behind 8528 holds no other ζ leaf; the whole orchestrator chain already waits on 5040 and
  8528 transitively through α → 9229. The fused-memory leaves form a second chain: η → ε2b → ε2a → ε1 →
  β2. β1 → β2 joins the two on `scripts/exception_register.py`, ε2a → δ takes the moved resolver, ι → α
  takes the house pattern. The old-name end state follows the last moves (D6): the operator gate σ → ζ5b,
  κ, χ (χ teaches the escalation watcher to spawn a pre-authorised gate's session); the refusal ξ → σ, η;
  and ω → ξ, ι gates the whole plan. π (the scoped-cap model spelling, D15) → κ, S20 ι. Wave-2 order is
  file by file (OQ-1, ruled by the lead).
- **D2 — Field defaults are the one home (R15); every survivor is the value production runs today.**
  `OrchestratorConfig()` always merges `defaults.yaml`, so the yaml value is the behaviour every
  operated project gets. Making it the field default changes nothing in production and changes only
  bare test constructions. The 23 disagreements at `1f240e6da2` (yaml / field → survivor):
  `max_concurrent_tasks` 24/3 → **24**; `lock_depth` 4/2 → **4**; `verify_cold_command_timeout_secs`
  5400/None → **5400**; `merge_verify_cold_command_timeout_secs` 7200/None → **7200**;
  `verify_clock_stop_enabled` True/False → **True**; `timeouts.reviewer` 1200/600 → **1200**;
  `routing.rules` the stock table / `[]` → **the stock table** (on `OrchestratorRoutingPolicy.rules`, C5);
  `models.implementer` sonnet/opus → **sonnet**; `models.debugger` sonnet/opus → **sonnet**;
  `models.reviewer` opus/sonnet → **opus**; `budgets.architect` 15/8 → **15**; `budgets.reviewer` 12/2 →
  **12**; `max_turns.architect` 180/75 → **180**; `max_turns.reviewer` 90/30 → **90**; `effort.architect`
  xhigh/high → **xhigh**; `effort.implementer` max/high → **max**; `effort.debugger` max/high → **max**;
  `effort.reviewer` xhigh/medium → **xhigh**; `effort.merger` max/high → **max**; `review.interval` 40/5 →
  **40**; `review.reviewer_briefing` `d2`/`diff-only` → **`d2`** (`ReviewBriefing.CONTEXT_COMPLETE`);
  `sandbox.enabled` False/True → **False**; `usage_cap.accounts_file`
  `${DARK_FACTORY_ROOT:…}/config/usage-accounts.yaml`/None → **the yaml leaf stays**. That last value
  is environment-layered: it is the one leaf `defaults.yaml` keeps, and the field default stays `None`.
  Rule for drift between authoring and α's dispatch: α takes the value `defaults.yaml` holds on α's base
  commit, never this table, because a wave-2 task may have changed one (S6 α 9140 retunes verify
  timeouts; 8375 adds the experiment-1 rules). α records the table it applied in its commit message.
  `sandbox.enabled` → False is called out because it flips a bare `SandboxConfig()` to unsandboxed.
  Keeping True would sandbox every operated project whose yaml does not set the key, which is a
  production behaviour change this stream was not asked to make.
- **D3 — `defaults.yaml` keeps only environment-layered leaves.** A leaf stays only if its value is a
  `${VAR}` / `${VAR:default}` interpolation and it names a declared field. Every other leaf is either
  deleted, when it restates the field default, or becomes the field default, when it disagrees (D2). The
  yaml's operator-facing comment prose moves to the field's `description` when it states a fact an
  operator needs, and to the commit message when it answers a reviewer (Comments stance). The guard (C7)
  walks every remaining leaf and fails on any leaf that is not such an interpolation, which covers a leaf
  that restates its default, one that disagrees with it, and one with no field (task 3038's reverse check).
  A default another package must read without importing the orchestrator lives in C8, and the field
  default names that constant (OQ-4, ruled by the lead: it reads R15's invariant as allowing a field
  default that names one shared constant).
- **D4 — The reload tier is a marker on the field; the reloadable set is derived (R15).**
  `Annotated[T, Reload.GREEN]` on a field makes it hot-reloadable. On a submodel-typed field it makes
  every leaf beneath it reloadable. On a `Mapping[StrEnum, BaseModel]`-typed field (a role table) it makes
  every row reloadable except the rows a `RestartOnlyRows` marker on the same field names, and
  `reloadable_paths` expands the table row by row. A row joins the GREEN set only when the leaf that moves
  its consumer shows the consumer resolves the value per call (boundary row 3). An unmarked field
  is restart-only. Every leaf that changes the reloadable set re-runs boundary row 6 against its base list
  and names each tier change in its commit: β1 first, then γ2, ε2a and ε2b, whose role tables move fields
  across tiers. `RELOADABLE_FIELDS` survives only as the derived constant
  `reloadable_paths(OrchestratorConfig)`, for the harness and the existing tests. A GREEN marker is a
  promise that every consumer reads the value at call time, never captured at construction. Each leaf
  that moves a GREEN field keeps that promise and states it in its own test (boundary row 3). The
  per-entry tier prose in today's allowlist moves to the field `description` when an operator needs it,
  otherwise to the commit message. `OPERATIONS.md` §"Config reload vs restart" keeps its group-level
  rules and points at the markers instead of a hand list (the 4559 comment records that the list lags).
- **D5 — Which fields become submodels.** Every field whose name encodes a subsystem prefix moves
  into that subsystem's submodel (R15: "per-subsystem submodels replacing the … prefix-encoded flat
  fields"). A moved field drops only the prefix its new submodel's name now carries; sub-prefixes stay
  (`scheduler.transient_requeue_cap`, `workflow.max_review_cycles`, `merge_lane.verify_breadth`). A field
  goes to the subsystem that executes the behaviour it tunes, never to the one that merely reads it.
  Project-level facts stay at the root: `project_root`, `test_command`, `lint_command`,
  `type_check_command`, `scope_cargo`, `module_overrides`, `max_per_module`, `env_overrides`,
  `role_env_overrides`, `strict_mcp_config`, `implementer_brief_variant`, `idle_poll_secs`,
  `invocation_timeout`, `orch_daily_cost_ceiling_usd`, `max_concurrent_tasks` and `lock_depth` (both
  project-level facts; the dashboard and fused-memory read the scheduler snapshot's published
  `lock_depth`, not the yaml). The remaining single-purpose root fields at `ba17934861`
  (`recycle_config_dir_on_zero_output`, `verify_env`, `prices`, `plan_text_fp_gate_enabled`,
  `simple_task_enabled`, `auto_eval_enabled`, `auto_eval_redo_budget_usd`, `auto_eval_phases`,
  `convert_to_blocked_enforce`, `verify_runners`, `verify_env_probes`) stay at the root in this plan.
  **The per-module layer keeps its flat spellings.** `ModuleConfig` is the per-module override schema,
  not a second spelling of root fields; ζ4b turns `_OVERRIDABLE_FIELDS` into
  `MODULE_OVERRIDE_TARGETS: Mapping[str, str]` (module key → root dotted path) beside `ModuleConfig`,
  with an import-time check that every target resolves in `OrchestratorConfig`. The ten module yamls are
  unchanged. A local, parameter or attribute that holds a submodel never has a name ending in `config` or
  `cfg` (write `sched`, `self._sched_knobs`); the reader-side completion checks rely on it. Each new submodel class lands in its own module under
  `orchestrator/src/orchestrator/config_sections/`, which `config.py` imports; the root field stays
  declared on `OrchestratorConfig` (OQ-9, ruled by the lead). `dashboard/orchestrator.yaml` is a module
  yaml, but it also carries six project-level keys no unit reads (`models`, `budgets`, `max_turns`,
  `max_execute_iterations`, `max_verify_attempts`, `max_review_cycles`); κ moves those to their new paths so
  the file would load if a unit ever ran it as a project config, and leaves its module-override keys flat.
  The submodels, by leaf:
  - ζ1: `service_restart: dict[RestartedService, ServiceRestartConfig]` (`fused_memory`, `dashboard`,
    `orchestrator`; enabled/debounce_secs/watch_prefixes/script/force_fire_after_secs, and the fused-memory
    row's restart-only `min_interval_secs`, carried from S24's `fused_memory_restart_min_interval_secs`) plus
    `fleet_redeploy` (the orchestrator-only min_interval/on_active/merge_phase_grace and
    `stale_service_restart_interval_secs`; S24 γ deletes `orchestrator_restart_lease_max_age_secs`); `watcher` (the 10 non-role `watcher_*` fields); `digest`;
    `escalation_revalidation`; `info_l0_router`.
  - ζ2: `recovery_sweeps`, one nested submodel per sweep: `orphan_l0`, `stranded`, `main_tip_sweep`,
    `terminal_status`, `no_landings_breaker`, `warm_lane_gc`, `worktree_orphan_reaper`,
    `deterministic_recon_sweep` (and task 4651's landed reconciler once it has landed), plus
    `lane_stale_report_days`.
  - ζ3: `scheduler` (age_alpha, cpm_beta, backfill_*, claimant dispatch gate and liveness TTL,
    dispatch cooldown, park_stop_*, pin_*, requeue_*, transient_requeue_*, snapshot interval,
    max_external_dep_unresolved_cycles, and the new `max_local_backfill_unresolved_cycles`), plus
    `delivered_checks.max_record_fetches_per_tick`.
  - ζ4a: `verify_admission`, `verify_clock_stop`, and `verify_cgroup` (the cpu weights and
    `use_cgroup_scope` join the existing memory submodel).
  - ζ4b: `verify` (command timeouts, cold commands, timeout retries, infra-retry knobs, drift-check
    cadence, concurrency, `sequential_lint_first`) and the module-override targets.
  - ζ5a: `merge_lane` (merge_train_*, heartbeat, history-gate knobs, disjoint-skip, advance attempts,
    escalate_preexisting_main_break, merge_verify_*).
  - ζ5b: `verify_hosts` (verify_host_*, verify_task_*, verify_cross_check_remote_green).
  - ζ6a: `workflow` (the iteration caps, review/judge cadence).
  - ζ6b: `workflow` gains rebase_*, revalidation_*, terminal cleanup grace, claimant heartbeat,
    exit-contract enforce and worktree identity guard; `steward` (steward_*, suggestion triage threshold).
  `GitConfig` (49 fields, 773 lines) is not grown: the merge lane's flat knobs get their own `merge_lane`
  submodel, and splitting `GitConfig` is out of scope.
- **D6 — Old setting names: translated while the moves land, then one sweep, then refused (ruling OQ-2 =
  B, Leo, 2026-11-12; R-5 = B, 2026-11-13).** Twelve project roots are known
  (`DASHBOARD_KNOWN_PROJECT_ROOTS`); nine run an orchestrator, all from this checkout's code, each reading
  its own yaml in its own repository, and the two remote-verify host yamls on leo-laptop are read by
  whichever unit's code dispatched the verify. A merged task reaches a unit only at its next start, so the
  fleet runs mixed code for days (three code shas across nine units on 2026-11-12). A yaml spelled only one
  way cannot suit both old and new code, so the end state is reached in three phases:
  1. **Translate (β1 to ζ5b, ε1 to η).** Every moved scalar field declares its old dotted spelling as a
     `LegacySpelling` marker; every move into a role table is declared on the table field as a
     `TableLegacy` (C3), so `models.<role>`, `budgets.<role>`, `unblock_auto.model`, `watcher_model`,
     `simple_task_budget_usd` and fused-memory's `reconciliation.sonnet_model` all pivot. A
     `mode='before'` root validator pivots old spellings to new paths, at load and at reload alike, in both
     `OrchestratorConfig` (β1) and `FusedMemoryConfig` (β2). Old and new present with different values is a
     load error naming both paths; the same value is accepted. The orchestrator's unknown-key census
     (`config.py::census_config_keys`) classifies a declared legacy spelling as `deprecated`, never
     `unknown`, so no level-2 escalation is raised; `check-config` and the reload report list the
     deprecations. Fused-memory, which has neither a `check-config` nor a census, lists them in its
     `reload_config` result and in a `config.deprecated_spellings` entry of its `get_status` result.
     `OrchestratorConfig.model_copy(update=…)` pivots a declared legacy key and raises on an unknown one
     (β1). This repository's own yamls are rewritten inside this plan's ordinary tasks: fused-memory's
     `config/config.yaml` inside each fused-memory moving task, and `dark-factory-orchestrator.yaml`,
     `dashboard/orchestrator.yaml` and the operator prose that names old keys in κ, after the last
     orchestrator move.
  2. **One pre-authorised sweep (σ).** After the last move is deployed to every running unit, one operator
     gate (cross-stream operator-sweep shape) raises a born-at-L2 escalation, and the escalation watcher
     spawns a /team session that Leo pre-authorised on 2026-11-12. The session rewrites the nine project
     yamls and the two leo-laptop host yamls, one yaml-only commit per repository. Every operated yaml then
     uses only new names, which every running code version knows. This follows R37's best-effort
     cross-repository edit, now as a task.
  3. **Refuse (ξ).** The translation path is deleted, and every declared old name becomes a load error and
     a hot-reload refusal. One error names every old name found in the file, each with its new path. The
     declarations stay: they are the refusal table. Because the sweep ran first, the refusal's deploy finds
     nothing to refuse, so no unit fails to start.

  Not `extra='forbid'`: an undeclared unknown key keeps today's census path (one self-closing level-2
  escalation through `harness.py::Harness._file_config_unknown_keys_escalation`), in every phase. The
  per-module layer's flat names are not old names (D5), so they are never refused. The `ORCH_` env source
  is neither pivoted nor refused: only `ORCH_PROJECT_ROOT` maps to a field today, and it does not move.
  The dated pivot-removal follow-up of the draft is replaced by ξ. Residual, stated: between κ's merge and
  the dark-factory unit's next restart, an operator `reload_config` on dark-factory run by code older than
  ζ5b would read κ's new spellings of ζ5b's settings as unknown keys. Those settings would fall back to
  their defaults until the restart, and the census's level-2 escalation would name each key, so the
  fallback is reported. κ's description says so. The lead confirmed κ with this residual (lead ruling after
  the wave-3 fold-verifiers).

  The same residual, quieter, applies to fused-memory's own `config/config.yaml`, which the ruling has
  rewritten inside each fused-memory moving task. Between ε2a's, ε2b's or η's merge and fused-memory's next
  restart (its on-merge restart runs at most every 4 hours), a fused-memory `reload_config` run by the older
  code reads the new spellings as nothing (fused-memory has no unknown-key census), and GREEN fields it no
  longer finds fall back to their defaults: the write-triage judge (pinned to `openai`/`gpt-6.1-sol`
  today) is the case that changes behaviour. Restart-only rows are reported `restart_required`, not
  applied. Each of those three leaves' descriptions says: no fused-memory `reload_config` between the merge
  and the restart. A restart never meets this window: it loads the new code and the new yaml together. The
  lead accepted this window as stated, the same shape as κ's residual (lead ruling after the wave-3
  fold-verifiers); closing it would need a fused-memory unknown-key refusal on reload, which is out of scope.

  After ξ, a new `LegacySpelling` refuses its old name on the deploy that ships it, so a later rename of a
  field any operated yaml sets would refuse that unit at its next start. Rule, stated in C3's docstring by
  ξ: from ξ on, a task that declares a new legacy spelling first runs `check-config` over the operated
  yamls (as ξ does); if any sets the old name, it does not declare the spelling but adds the new field and
  leaves the rename to a later translate, sweep and refuse cycle. Pending task 6754 renames a `GitConfig`
  field that reify's yaml sets: σ → 6754 puts it inside this cycle, so σ's sweep covers it.
- **D7 — One role table (R15).** `OrchestratorConfig.roles: dict[RoleName, RoleSettings]` is complete
  over `ORCHESTRATOR_ROLES`: today's 11 config roles plus `UNBLOCK_AUTO` and `WATCHER`. Its
  `default_factory` is the one home of every role default (D2's survivors included). A project yaml
  entry overrides only the knobs it names, and every unnamed knob keeps that role's default: a
  `mode='before'` validator merges entry-wise and knob-wise through `merge_role_table`, after the legacy
  pivot has placed any knob-major spelling, because pydantic would otherwise replace a dict wholesale.
  An unknown role key is a load error. `unblock_auto.enabled` and its B3 knobs stay in `unblock_auto`, and
  the watcher's supervisor knobs move to `watcher` (ζ1). `startup_grace_secs` and `working_idle_secs`
  leave the role namespace for an `agent_liveness` submodel. The deprecated
  `simple_task_budget_usd`/`simple_task_max_turns` scalars and their migrator become legacy spellings
  of `roles.simple_task.*` (D6). Allowlist validation runs once over the table, scoped to
  `Backend.CLAUDE` rows: a non-Claude model namespace is not the Claude-alias allowlist's business, as
  today. That scope now includes the watcher.
- **D8 — `config_role` is the reviewer collapse, declared once.** `shared.roles.config_role(name)` maps
  a dispatch identity to its config role. An exact `RoleName` value maps to itself, a member of
  `REVIEWER_VARIANTS` or the `reviewer_<name>` family maps to `REVIEWER`, and any other name raises
  `UnknownRole` naming the value. A parity test asserts it is total over `orchestrator.agents.roles.ROLES`.
  `shared.task_metadata.KNOWN_ROLE_NAMES` is derived as
  `{r.value for r in DISPATCHABLE_ROLES} | REVIEWER_VARIANTS`, which keeps today's 12 names (the S10
  wire contract), and the orchestrator parity test asserts `set(ROLES) ⊆ KNOWN_ROLE_NAMES`. Layer 1 of
  the resolver (a task's `model_overrides` pin) keeps keying on the full dispatch identity, never on
  `config_role(name)`, so the CAVEAT on `KNOWN_ROLE_NAMES` (collapsed keys are accepted but inert) still
  describes the behaviour. `routing.py::_config_key` and the inline `startswith` in `TaskWorkflow._invoke`
  are deleted.
- **D9 — The resolver moves to shared; the orchestrator keeps only its own post-transform (R15).**
  `shared.route_policy` holds the policy models (`RoutingPolicy`, `RoutingRule`, `RuleMatch`, `RuleSet`
  and `SplitMatch` once task 8373 lands) and `validate_role_models`. Their field defaults are the
  defaults that were `DEFAULT_ALLOWED_MODELS`/`DEFAULT_LADDER`, so the TYPE_CHECKING cycle and both
  constants go. The orchestrator's stock rule table is the `rules` default of
  `orchestrator.config.OrchestratorRoutingPolicy(RoutingPolicy)`, the type of `OrchestratorConfig.routing`,
  so a project yaml with a partial `routing:` block (say, only `allowed_models`) keeps the stock rules as it
  does today through the yaml deep-merge; fused-memory's `model_routing` stays a plain `RoutingPolicy`.
  `shared.route_resolver.resolve_route` takes `(inputs, policy, settings)`. The layer semantics are
  byte-identical: precedence, ladder bumps, fail-safe rejections recorded as
  `shared.route_rejection.RouteRejection` values (S25 C4), and the claude-backend scope. The decision
  also carries `backend` and `timeout_secs` from the role row, which no layer overrides, so
  `TaskWorkflow._invoke` stops reading them itself. `implementer_brief_variant`'s
  `invocation_bounds` is orchestrator policy, so `routing_dispatch` applies it after `resolve_route`.
  `orchestrator/routing.py` keeps only the availability probe (`probe_models`,
  `classify_probe_outcome`, `render_probe_artifact`), which imports `orchestrator.config` downward at
  runtime. The cost-ceiling fail-closed semantics from task 5366 are kept as that task lands them.
  `validate_role_models` also refuses a `Backend.CLAUDE` row whose `budget_usd` is below S20's
  `INVOCATION_COST_FLOOR_USD` (S20 C6), which every Claude invocation would otherwise refuse at dispatch.
- **D10 — Fused-memory roles and providers (R15).** Fused-memory's config gains
  `model_routing: RoutingPolicy`. The name avoids its existing write/read-classification `routing:
  RoutingConfig`. It also gains `roles: dict[RoleName, RoleSettings]`, complete over
  `FUSED_MEMORY_ROLES`: `MEMORY_LLM` (`llm.model`/`llm.provider`; the block drives Graphiti extraction,
  Mem0's LLM and the write/read classifier), `RECON_AGENT`, `RECON_JUDGE`, `RECON_STAGE_STANDARD`
  (`sonnet_model`), `RECON_STAGE_BACKLOG` (`opus_model`), `CURATOR`, `PATH_SCOPE_ADJUDICATOR`,
  `WRITE_TRIAGE_JUDGE` and `LINK_ADJUDICATOR` (names as drafted, OQ-7, ruled by the lead). Every old field becomes a legacy spelling,
  declared as a `TableLegacy` row map on `roles` (D6). `'claude_cli'` parses to `Backend.CLAUDE`. The recon
  tiers keep their `episode_limit`/`memory_limit` in `ReconciliationConfig`; only the model moves. Every
  fused-memory default row sets `model`, `effort`, `budget_usd` and `max_turns`, so the row is the
  resolver's layer-4 `RoleDefaults` (C6): for a single-call SDK role that is `max_turns=1` and the effort
  the call already sends (the write-triage judge's `judge_reasoning_effort`). The write-triage judge's
  inheritance from the memory LLM (`resolve_judge_provider`/`resolve_judge_model`) is retired (OQ-8, ruled by the lead, flagged): its
  row has its own default, today's effective values, because the live config pins the judge explicitly
  (`openai` / `gpt-6.1-sol`) and a cross-row rule inside the table merge is a second mechanism (heuristic
  3); an operator who retunes the memory LLM sets the judge explicitly. The table keeps today's reload tiers
  row by row (D4): `WRITE_TRIAGE_JUDGE` and `LINK_ADJUDICATOR` are GREEN, as `write_triage.judge_*` and
  `link_heal.adjudicator_model` are today; `RestartOnlyRows` names `MEMORY_LLM`, the four recon rows,
  `CURATOR` and `PATH_SCOPE_ADJUDICATOR`, which are restart-only today, and
  `backends/graphiti_client.py::build_llm_client` and `backends/mem0_client.py` fix the memory model at
  construction. A row becomes GREEN only when the leaf that moves its consumer shows the consumer resolves
  per call (boundary row 3). `LLMConfig`'s validator cross-checks `client_class` and
  `structured_output_mode` against the provider; that validation moves with the provider to a
  `FusedMemoryConfig` validator over the `MEMORY_LLM` row (ε2b). `cli_stage_runner`'s `max_budget_usd=5.0` becomes the stage
  role's `budget_usd`. Admitting a model beyond the stock allowlist still takes one edit per service
  (each project's `routing.allowed_models` and fused-memory's `model_routing.allowed_models`) on one
  shared default (OQ-10, ruled by the lead); this plan states that cost rather than adding a third config source.
- **D11 — Recon providers lose their dead arms (R15; OQ-3, ruled by the lead, flagged).** `ReconciliationConfig.agent_llm_provider`
  and `judge_llm_provider` become `Backend` legacy spellings of `roles.recon_agent.backend` and
  `roles.recon_judge.backend`, restricted to `{CLAUDE}` for the agent loop. AgentLoop's anthropic and
  openai paths go: `_call_llm`'s SDK arms, `_call_openai`, `_OpenAIResponseAdapter` and the
  `_TextBlock`/`_ToolUseBlock` shims, which only tests reach. The judge's `else` → OpenAI arm goes. Under
  the recommendation, the judge's anthropic arm and the prose verdict path (`Judge._parse_verdict`'s text
  arm, and the hand-written "## Output Format" JSON in `reconciliation/prompts/judge.py`) go too, leaving
  `JUDGE_VERDICT_SCHEMA` as the verdict contract's one home. The lead ruled the recommendation and flagged
  its consequence: the SDK provider stops being a valid judge setting, which extends R15's deletion.
- **D12 — Service-path knobs leave `ReconciliationConfig`; consumers trust the schema (R15).** η moves
  every `ReconciliationConfig` field whose only readers are outside `reconciliation/` into a server-owned
  section on `FusedMemoryConfig`: `ProceduralGuardConfig` (the five `procedural_knowledge_*` fields),
  `SearchAnchorConfig` (`topic_anchored_recall_enabled`), and the provenance authz prefixes into the
  section their middleware owns. Each moved field carries a `LegacySpelling` of its
  `reconciliation.*` path. `_default_topic_guard_clusters`' seed data moves to its own module.
  Consumers read the typed field directly. The three `_reconciliation_attr` copies, the
  `isinstance`/`getattr` fallbacks in the content-amend and entity-mint storm detectors (now in S26's
  `MemoryMutations`/`GraphMutations`), and the restated defaults `_DEFAULT_TOPIC_ANCHOR_ENABLED` and the
  `near_duplicate_guard` default mirror are deleted. Tests that fed attribute-free doubles build a real
  `FusedMemoryConfig` through a typed test factory (the Tests stance).
- **D13 — The two double-duty knobs get their own fields (R15).**
  `delivered_checks.max_record_fetches_per_tick` bounds `_phase_backfill_terminal_dep_records`' fetch
  fan-out. `scheduler.max_local_backfill_unresolved_cycles` sets `_phase_backfill_dep_status`'
  degradation warning threshold. Each defaults to today's shared value and takes its sibling's tier.
  Each description states the one behaviour it governs, and the scheduler comments about the deferral
  are deleted.
- **D14 — Prose stance.** Every docstring, comment and description a leaf rewrites follows the
  Comments stance. Moved rationale, measurements and rejected alternatives go to commit messages, and
  PRD letters are never written into code.
- **D15 — The scoped-cap model has one spelling (lead decision on the task-8253 investigation's side
  finding).** The fact "which model has its own per-account cap scope" is spelled three ways today:
  `fable` (the CLI alias the nightly preparer and the census pass), `claude-fable-5` (the
  `scoped_cap_models` default, which no production path dispatches; only two frozen eval arms do,
  `evals/configs.py`'s `architect-fable-high` and `architect-fable-max`) and `[claude-fable-5, claude-fable-5-1]`
  (dark-factory's yaml). So `scope_for('fable')` is `None`, and a Fable cap in a one-shot pool marks the
  whole account capped. π makes C9 (`shared/model_names.py`) the one home: `FABLE_MODEL` is the admitted
  Fable id, and `canonical_model` is the one decoder of a CLI alias to it. The `scoped_cap_models` field
  default and `orchestrator/routing.py`'s admitted-Fable constant name `FABLE_MODEL`, `scope_for` matches
  `canonical_model(model)`, and dark-factory's yaml override goes because it would restate the default.
  The preparer and the census keep passing the alias, which the decoder handles; admission
  (`routing.allowed_models`, the per-model ceiling) stays a per-project operator decision. Admitting a new
  Fable is then one edit to `FABLE_MODEL`, which moves scoping, the availability probe and the alias
  together. Impact today is near nil: all 801 weekly caps in 30 days were account-wide anyway
  (`digests/leo-rulings-8253.md`). Two effects an operator could notice, stated (the lead told Leo; lead
  ruling after the wave-3 fold-verifiers): (1) the one-shot pools
  (`scripts/legibility/account_pool.py::_pool_config` inherits the field default) start scoping Fable caps,
  so a Fable cap in the nightly preparer stops only Fable work on that account instead of the whole
  account; (2) `claude-fable-5` leaves the scoped set (today it is in the default and in dark-factory's
  list), so a cap hit by the two frozen `claude-fable-5` eval arms caps that eval gate's whole account
  rather than only its Fable scope. Admission is untouched: `claude-fable-5` stays out of the stock
  allowlist (CLAUDE.md §"Model Routing") and so does `FABLE_MODEL`; C9 names the scoped model, it admits
  nothing.
- **Standing rules of 2026-11-12.** *`one-decision-partition-per-area`* (INV-23, minted by S17): this plan's one procedure
  table is σ's step table, and each row names who decides it. Every other decision in the plan is code:
  translate or refuse (D6), the allowlist and cost-floor refusal (C5), and the reload tier (D4). No row
  that code can decide is handed to an LLM. *Hermetic evals*: this plan adds no store, sink, filer, queue
  or archive. `evals/runner.py::build_eval_orch_config` gains role rows (γ2) and still passes no production
  filer, queue, event store or cost store.

## Contracts

S16 owns C1–C9 (G4 row: "shared routing resolver + allowlist; shared hot-reload engine;
`RoleName`/`Backend` enums; field defaults as the one home"). A dependent stream holds a leaf on
`S16/C<n>`. The fold added C8 and widened C2, C3, C5, C6 and C7 in place. The ruling fold widened C1 (two
members) and C3 (the refusal, landing in ξ) in place and added C9; C1–C8 keep their numbers.

**C1 — `shared/src/shared/backends.py` (lands in γ0; S20 consumes for `invocation_outcome` and
`cli_invoke`).**
```python
class Backend(StrEnum):
    CLAUDE = 'claude'          # the Claude Code CLI (shared.cli_invoke)
    CODEX = 'codex'; GEMINI = 'gemini'; PI = 'pi'
    ZCODE = 'zcode'; MIMO = 'mimo'   # landed by tasks 5399 and 7993; outside the account pool
    ANTHROPIC_API = 'anthropic'   # in-process Anthropic SDK
    OPENAI_API = 'openai'         # in-process OpenAI-compatible SDK
AGENT_CLI_BACKENDS: Final[frozenset[Backend]]   # what orchestrator.agents.invoke dispatches
SDK_BACKENDS: Final[frozenset[Backend]]
def parse_backend(value: str | Backend) -> Backend: ...   # the one decoder; 'claude_cli' -> CLAUDE
class UnknownBackend(ValueError): value: str; allowed: tuple[str, ...]
```
Invariants: stdlib-only. `parse_backend` is the only place a backend string is decoded, and the one
legacy alias is `'claude_cli'`. An unknown value raises `UnknownBackend`, whose message names the
value and the allowed set. Every pydantic field of backend type is annotated `Backend`, with a
`mode='before'` hook that calls `parse_backend`. `AGENT_CLI_BACKENDS` is the six CLI members
(`CLAUDE`, `CODEX`, `GEMINI`, `PI`, `ZCODE`, `MIMO`), exactly the arms `invoke_agent` dispatches at
`ce6673c7e1`; a test asserts the dispatch table is total over it. Which members are pooled
(`ACCOUNT_POOL_BACKENDS`) is S20's to type on these members.

**C2 — `shared/src/shared/roles.py` (lands in γ0; consumed by γ1, γ2, δ, ε1, ε2a/b).**
```python
class RoleName(StrEnum): ...   # config roles: ARCHITECT IMPLEMENTER DEBUGGER REVIEWER MERGER STEWARD TRIAGE
                               # MODULE_TAGGER DEEP_REVIEWER JUDGE SIMPLE_TASK UNBLOCK_AUTO WATCHER
                               # + MEMORY_LLM RECON_AGENT RECON_JUDGE RECON_STAGE_STANDARD RECON_STAGE_BACKLOG
                               #   CURATOR PATH_SCOPE_ADJUDICATOR WRITE_TRIAGE_JUDGE LINK_ADJUDICATOR   (OQ-7, lead)
ORCHESTRATOR_ROLES: Final[frozenset[RoleName]]
DISPATCHABLE_ROLES: Final[frozenset[RoleName]]          # ORCHESTRATOR_ROLES - {UNBLOCK_AUTO, WATCHER}
FUSED_MEMORY_ROLES: Final[frozenset[RoleName]]
REVIEWER_VARIANTS: Final[frozenset[str]] = frozenset({'reviewer_comprehensive'})   # dispatch identities
def config_role(dispatch_name: str) -> RoleName: ...   # D8; raises UnknownRole(value)
class RoleSettings(BaseModel):   # frozen, extra='forbid'
    model: str
    backend: Backend = Backend.CLAUDE
    effort: str | None = None         # γ0 decides (tactical 3): a closed union of the claude, codex and pi levels, or str
    budget_usd: float | None = None   # > 0 when set
    max_turns: int | None = None      # > 0 when set
    timeout_secs: float | None = None # > 0 when set
def merge_role_table(defaults: Mapping[RoleName, RoleSettings], raw: Mapping[str, Mapping]) -> dict[RoleName, RoleSettings]: ...
```
Invariants: `config_role` is total over the orchestrator's dispatch identities, enforced by a test in
`orchestrator/tests`; `config_role` consumes `REVIEWER_VARIANTS`, so `shared` never imports
`orchestrator.agents.roles`. `merge_role_table` is the one entry-wise, knob-wise merge (D7). It raises on
an unknown role key or knob and never drops one. `RoleSettings` is immutable; a reload replaces the row
inside its table, and the table object keeps its identity (C4). No effort vocabulary exists in the tree
today (`EffortConfig` fields are `str`; pi's levels live in `agents/invoke.py::_PI_THINKING_LEVELS`), so γ0
either defines the union or keeps `str`.

**C3 — `shared/src/shared/config_fields.py` (lands in β0).**
```python
class Reload(Enum): GREEN = 'green'                         # Annotated marker; absent = restart-only
@dataclass(frozen=True) class LegacySpelling: path: str     # an old dotted spelling of the field it annotates
@dataclass(frozen=True) class TableLegacy:                  # on a role-table field: moves INTO the table
    rows: Mapping[str, tuple[str, str]]                     # old dotted path -> (row key, knob)
    @classmethod
    def knob_major(cls, old_block: str, knob: str) -> 'TableLegacy': ...   # '<old_block>.<row>' -> (row, knob), every row of the key enum
@dataclass(frozen=True) class RestartOnlyRows:              # on a GREEN role-table field: these rows stay restart-only
    rows: frozenset[str]
@dataclass(frozen=True) class DeprecatedSpelling: old_path: str; new_path: str
class ConflictingSpellings(ValueError): old_path: str; new_path: str
def reloadable_paths(model_cls: type[BaseModel]) -> frozenset[str]: ...
def legacy_spellings(model_cls: type[BaseModel]) -> Mapping[str, str]: ...   # old_path -> new_path, derived, both forms expanded
def pivot_legacy_spellings(model_cls: type[BaseModel], data: Mapping) -> tuple[dict, tuple[DeprecatedSpelling, ...]]: ...
# the end state, landing in ξ (the translation phase of D6 ends):
class LegacySpellingRefused(ValueError): spellings: tuple[tuple[str, str], ...]   # every (old_path, new_path) found
def refuse_legacy_spellings(model_cls: type[BaseModel], data: Mapping) -> None: ...   # replaces pivot_legacy_spellings
```
Invariants: both derivations walk declared annotations only (`BaseModel`, `Optional[BaseModel]` and
`Mapping[StrEnum, BaseModel]` are descended), so a tier or a spelling is stated once, on its field.
`reloadable_paths` expands a GREEN table field row by row and leaves out every row its `RestartOnlyRows`
names. The
table pivot is declared on the container field (the `roles` field of each config), and its new path is
`<table field path>.<row>.<knob>`; `knob_major` expands over the field's declared key enum, so
`TableLegacy.knob_major('models', 'model')` maps `models.architect` to `roles.architect.model`.
`legacy_spellings()` expands both forms into old_path → new_path pairs, and the `ConflictingSpellings`
rule and the `DeprecatedSpelling` report apply to every expanded pair. `pivot_legacy_spellings` runs
before any table merge (`merge_role_table`) and is pure. It never drops a key: a key it cannot place is
left for the unknown-key census. It raises `ConflictingSpellings` naming both paths when old and new
disagree. Two fields declaring the same legacy path fail at class creation (import), not at load. A
marker on a class both services use (`UsageCapConfig`, `RoleSettings`, `RoutingPolicy`) applies in both
services; a per-service tier or spelling goes on the container field. From ξ on, both configs' root
validators call `refuse_legacy_spellings` instead of `pivot_legacy_spellings`: it walks the same derived
`legacy_spellings` map, and when any declared old path is present it raises one `LegacySpellingRefused`
whose message lists every old path found with its new path. It never inspects an undeclared key. ξ deletes
`pivot_legacy_spellings`, `DeprecatedSpelling` and `ConflictingSpellings`, which nothing can reach once
old names are refused. The `LegacySpelling` and `TableLegacy` markers stay as the refusal table, and the
map is also what σ's session reads.

**C4 — `shared/src/shared/config_reload.py` (lands in β0; orchestrator adopts in β1, fused-memory in
β2).**
```python
@dataclass(frozen=True) class ConfigDiff: applied_candidates: Mapping[str, Change]; restart_required: Mapping[str, Change]; unchanged: int
def iter_leaves(model_cls: type[BaseModel], model: BaseModel | None) -> Iterator[tuple[str, object]]: ...
def diff_config(live: BaseModel, fresh: BaseModel, reloadable: frozenset[str]) -> ConfigDiff: ...
def apply_reload(live: M, fresh: M, *, reloadable: frozenset[str] | None = None) -> ReloadReport: ...
# ReloadReport = {reloaded, applied, restart_required, unchanged, error}  (today's shape, both services)
#   + deprecated: list of {old_path, new_path} while D6's translation phase lasts; ξ removes the key
```
Invariants: descent follows the declared annotation. A `None` side yields the `_ABSENT` sentinel, and
the diff iterates the union of both sides' paths: the esc-2718-1 fix, now for both services.
`reloadable` defaults to `reloadable_paths(type(live))`. Apply writes in place and preserves the identity
of every submodel and table object a consumer holds, so a consumer that captured `config.scheduler`
reads the new values. After applying, it re-validates `type(live)` as a whole and rolls every applied leaf
back on failure, so a failed reload leaves `live` exactly as it was. Pure apart from the in-place write:
no I/O and no logging. The callers (the harness's `reload_config` and fused-memory's `reload_config`
tool) keep their audit and thread-off loading.

**C5 — `shared/src/shared/route_policy.py` (lands in δ; fused-memory adopts in ε2a).**
```python
class RuleMatch / RuleSet / RoutingRule / SplitMatch   # moved verbatim from orchestrator.config (8373's SplitMatch included)
class RoutingPolicy(BaseModel):   # extra='forbid'
    allowed_models: list[str] = Field(default_factory=lambda: ['haiku', 'sonnet', 'opus'])
    ladder: list[str] = Field(default_factory=lambda: ['haiku', 'sonnet', 'opus'])
    per_model_daily_ceiling_usd: dict[str, float] = Field(default_factory=dict)
    rules: list[RoutingRule] = Field(default_factory=list)
def validate_role_models(roles: Mapping[RoleName, RoleSettings], policy: RoutingPolicy) -> None: ...
# orchestrator/src/orchestrator/config.py (lands in δ):
class OrchestratorRoutingPolicy(RoutingPolicy):
    rules: list[RoutingRule] = Field(default_factory=lambda: list(_STOCK_RULES))   # the D2 survivor
```
Invariants: each default is stated once. The orchestrator's stock rule table is the `rules` default of
`OrchestratorRoutingPolicy` (D2), the type of `OrchestratorConfig.routing`, never of `RoutingPolicy.rules`,
so a partial `routing:` block keeps the stock rules (boundary row 10). `validate_role_models` checks every
`Backend.CLAUDE` row's model against `allowed_models` and its `budget_usd` against
`shared.claude_argv.INVOCATION_COST_FLOOR_USD` (S20 C6), and raises `ValueError` naming the role, the
value and the allowlist or floor. Both configs call it from a model validator, so load and reload enforce
it alike.

**C6 — `shared/src/shared/route_resolver.py` (lands in δ; fused-memory adopts in ε2a/ε2b).**
```python
@dataclass(frozen=True) class PlanShape: step_count: int; module_paths: tuple[str, ...]
@dataclass(frozen=True) class RoleDefaults: model: str; effort: str; budget_usd: float; max_turns: int
@dataclass(frozen=True) class RouteInputs: role_name: str; task_id: str | None; task_metadata: Mapping[str, object]
    plan_shape: PlanShape | None; routing_tier: int; dispatch_count: int; role_defaults: RoleDefaults
    spend_by_model: <as task 5366 lands it>; scope_capacity: Mapping[str, bool] | None = None
    @classmethod
    def for_role(cls, role_name: str, role_defaults: RoleDefaults) -> 'RouteInputs': ...
        # task_id None, empty metadata, no plan shape, tier 0, dispatch 0: a call that is not a task dispatch
@dataclass(frozen=True) class RoutingDecision: model: str; effort: str; budget_usd: float; max_turns: int
    backend: Backend; timeout_secs: float | None; source_layer: str; rule_id: str | None
    rejections: tuple[RouteRejection, ...]          # S25 C4
    @property rejected -> tuple[str, ...]           # rendered, for the existing event and mirror fields
def resolve_route(inputs: RouteInputs, policy: RoutingPolicy, settings: RoleSettings) -> RoutingDecision: ...
```
Invariants: total. It never raises, because layer 4 (`role_defaults`) is unconditional. It is pure.
The layer order and rejection semantics are byte-identical to `orchestrator.routing.resolve_route` at
δ's base (boundary row 10). The settings row is the caller's `config.roles[config_role(role_name)]`.
`backend` and `timeout_secs` come from that row unmodified. A fused-memory caller builds
`RouteInputs.for_role(role, defaults)` with `defaults` taken from its role's default row (model, effort,
budget_usd, max_turns), which sets all four knobs (D10); the orchestrator's dispatch path keeps building the full inputs from the task.

**C7 — the config-default-one-home guard (lands in α; promoted in ι).** A test,
`orchestrator/tests/test_config_default_one_home.py`, walks every `defaults.yaml` leaf against
`OrchestratorConfig.model_fields`, following submodels and role tables. It asserts that every remaining
leaf names a declared field and that its value is a `${…}` interpolation. That one assertion covers a
leaf that restates its default, one that disagrees with it, and one with no field. Each offending path is
named with its reason. It is exercised against three seeded fixture yamls (a restating leaf, a leaf with
no field, a disagreeing literal leaf; INV-10) and reads no prose.

**C8 — `shared/src/shared/config_defaults.py`, cross-package defaults (lands in α; OQ-4 option 2, ruled
by the lead).**
```python
DEFAULT_MAX_CONCURRENT_TASKS: Final[int] = 24
DEFAULT_LOCK_DEPTH: Final[int] = 4
```
Invariants: stdlib-only, and holds only defaults that a package which must not import the orchestrator
reads (today the dashboard; S22 λ adds the escalation port). Each constant is the default of exactly one
config field, written `Field(default=DEFAULT_…)`, and no other literal of that default exists. α creates
the module. S24's `shared/fleet_coordination.py` is the same pattern and stays S24's: it homes
`WATCH_CLOSURE` (the default of both watch-prefix fields) and `FM_ON_MERGE_MIN_INTERVAL_SECS` (the default
of the restart-only `fused_memory_restart_min_interval_secs` S24 ζ adds; S24 owns the value, 14400 until S24
θ flips it to 3600 after operator gate 9521 puts task 9493's stop fix in effect). The invariant treats a
field default naming one shared constant as one home. No S16 task edits `shared/fleet_coordination.py`, and
α takes no edge onto S24 (lead ruling after the wave-3 fold-verifiers).

**C9 — `shared/src/shared/model_names.py`, the scoped-cap model's one spelling (lands in π; lead decision
on the task-8253 side finding).**
```python
FABLE_MODEL: Final[str] = 'claude-fable-5-1'                  # the admitted Fable id
MODEL_ALIASES: Final[Mapping[str, str]]                        # CLI alias -> canonical id: {'fable': FABLE_MODEL}
def canonical_model(name: str) -> str: ...                     # the one decoder; an unlisted name passes through
```
Invariants: stdlib-only. `canonical_model` is the only place a model alias is decoded for cap scoping, and
it is total: an unlisted name is returned unchanged, so `haiku`/`sonnet`/`opus` routing aliases are
untouched. `UsageCapConfig.scoped_cap_models` defaults to `[FABLE_MODEL]`, `scope_for(model, config)`
matches `canonical_model(model)`, and `orchestrator/routing.py`'s admitted-Fable constant names
`FABLE_MODEL`. Scope keys stay canonical model ids, so the resolver's `scope_capacity` lookups, the `scope`
detail of cost events and `scripts/audit_model_admission.py` keep their shape.

## Boundary-test sketch

| # | Side | Scenario | Pre | Post |
|---|---|---|---|---|
| 1 | shared (β0) | reload a model whose `Optional` submodel goes `None` → populated and back | fixture model, both services' shapes | diff lists the leaf paths; no `KeyError`; apply round-trips |
| 2 | shared (β0) | add a field to a GREEN submodel of a fixture model | no other edit | the new leaf is in `reloadable_paths`; an unmarked root field is not |
| 3 | orchestrator (β1, every ζ) | reload changes a GREEN field a consumer holds by reference | live harness test double over the real config object | the next call reads the new value; submodel identity unchanged |
| 4 | shared (β0) | pivot `{old: 6}`; `{old: 6, new: 6}`; `{old: 6, new: 7}` | fixture with one `LegacySpelling` | value at new path + one `DeprecatedSpelling`; same; `ConflictingSpellings` naming both |
| 4b | shared (β0) | pivot `{models: {architect: opus}}`; `{unblock_auto: {model: opus}}`; `{models: {architect: opus}, roles: {architect: {model: sonnet}}}` | fixture with a `dict[StrEnum, BaseModel]` table carrying `TableLegacy.knob_major` and a row map | `roles.architect.model == 'opus'` + one deprecation; the unblock row likewise; `ConflictingSpellings` naming both |
| 5 | orchestrator (β1) | `census_config_keys` on a yaml using a declared legacy spelling | fixture yaml | reported `deprecated`, not `unknown`; no born-at-L2 filing |
| 6 | orchestrator (β1; re-run by γ2, ε2a and ε2b) | derived reloadable set vs the leaf's base list | the base list frozen as a fixture (β1: today's hand list) | equal, or each difference named in the commit as a deliberate tier decision, role-table rows included |
| 7 | shared (γ0) | `parse_backend('Claude')`, `('claude_cli')`, `('zcode')`, `('mimo')`; `config_role('reviewer_comprehensive')`, `('nope')` | — | `UnknownBackend` naming the allowed set; `CLAUDE`; `ZCODE`; `MIMO`; `REVIEWER`; `UnknownRole` |
| 8 | orchestrator (γ1) | load and reload a yaml with `backends.reviewer: Claude` | live-shaped yaml | load raises naming the field; reload returns `reloaded: false` with the error |
| 9 | orchestrator (γ2) | `roles: {architect: {model: opus}}`; unknown role key; `roles.watcher.model: gpt-x` | defaults table | architect keeps its default budget/turns/effort; unknown key raises; watcher model outside the allowlist raises at load |
| 9b | orchestrator (γ2) | reify's committed yaml copy before and after γ2 | knob-major `budgets`/`max_turns`/`effort`/`timeouts` blocks | the same resolved values; each block reported deprecated, none unknown; `set(ROLES) ⊆ KNOWN_ROLE_NAMES` |
| 10 | shared ↔ orchestrator (δ) | the routing byte-equivalence corpus (`test_routing_byte_equivalence.py`) through the shared resolver, plus an `implementer_brief_variant` case run through `routing_dispatch.py::resolve_and_record_route`, plus a project yaml whose `routing:` block names only `allowed_models` | stock rule table, overrides, ceilings, capacity snapshots | identical model, effort, budget, turns, source_layer, rule_id and rendered `rejected` to the pre-move resolver; the variant's bounds applied after resolution; the partial block resolves the stock table |
| 11 | orchestrator (δ) | dispatch `reviewer_comprehensive` | role table | decision's `backend`/`timeout_secs` are the reviewer row's; no `startswith('reviewer')` anywhere |
| 12 | fused-memory (ε1) | config with `judge_llm_provider: claude-cli` (typo); `claude_cli` | — | load raises `UnknownBackend`; `claude_cli` loads as `CLAUDE` |
| 13 | fused-memory (ε2a) | live `fused-memory/config/config.yaml`; a recon role with a model outside `model_routing.allowed_models`; a Claude role with `budget_usd` below S20's floor | — | live loads, values equal to today's; the bad model and the bad budget each refuse naming role, value and allowlist or floor; a recon stage invocation's `max_budget_usd` is the role's |
| 13b | fused-memory (ε2a) | a fused-memory yaml with `reconciliation.sonnet_model: opus` | fixture config | loads with `roles.recon_stage_standard.model == 'opus'` and one deprecation, listed by `reload_config` and `get_status` |
| 14 | fused-memory (ε2b) | write-triage judge route for the live config; reload `roles.write_triage_judge.model`; reload `roles.memory_llm.model` | live config | provider `openai`, model `gpt-6.1-sol` as today; the next judge call uses the new model; the memory-LLM reload reports `restart_required`, not `applied` |
| 15 | fused-memory (η) | legacy `reconciliation.topic_anchored_recall_enabled`; `mem0_update.storm_threshold: 0` | fixture configs | pivoted + one deprecation; load raises (schema `gt=0`), with no consumer fallback branch left to reach |
| 16 | orchestrator (α) | the guard over three seeded fixture yamls and over the live `defaults.yaml` | one restating leaf; one leaf with no field; one disagreeing literal leaf | each fixture flagged with its path and reason; live file: only `${…}` leaves remain |
| 17 | orchestrator (ζ3) | raise `max_checks_per_tick`; lower `max_external_dep_unresolved_cycles` | scheduler test builder | the record-fetch fan-out and the local-backfill warning threshold do not change |
| 18 | all (every ζ, κ; ξ flips it) | resolve copies of the dark-factory and reify yamls before and after | committed fixture copies plus the old→new path map | while translation lives: every moved value equal at its new path, and dark-factory's fixture after κ has zero deprecations; from ξ on: the old-spelled copies are refused (row 22) and the rewritten copies resolve to the same values |
| 18b | orchestrator (β1) | `config.model_copy(update={'<declared legacy key>': v})`; `model_copy(update={'no_such_knob': 7})` | real `OrchestratorConfig` | the value lands at the new path, the copied submodel's other knobs keep their values, and the original config is unchanged (`model_copy` carries submodels by reference); the unknown key raises naming it |
| 19 | orchestrator (ζ4b) | a module yaml setting `verify_command_timeout_secs`; a `MODULE_OVERRIDE_TARGETS` entry whose target does not resolve | fixture module tree | the module's verify uses its value over `verify.command_timeout_secs`; the bad target fails at import |
| 20 | orchestrator (ζ4b, ζ5a) | a remote-verify spec carrying `merge_verify_breadth: full` and `verify_cold_preprovision_command` applied to a host config through `verify_spec_shared.py::apply_merge_verify_spec` (S4 η 9135); dark-factory's yaml through `merge_lane/gates.py` | committed spec fixture | both land at their nested paths in the `AppliedSpec`'s config; the spec's dict keys are unchanged; the merge gate is whole-tree |
| 21 | shared (ξ) | `refuse_legacy_spellings` on `{old: 6}`; on `{old_a: 6, old_b: 'x'}`; on `{new: 6}`; on `{undeclared: 1}` | fixture with two `LegacySpelling`s and one `TableLegacy` | one `LegacySpellingRefused` naming `old → new`; one error naming both pairs; no error; no error (the census path owns undeclared keys) |
| 22 | orchestrator (ξ) | load and `reload_config` the committed old-spelled dark-factory and reify fixture copies; `check-config` on them; `config.model_copy(update={'<declared old key>': v})` | row 18's fixtures | load raises naming every old key with its new path; reload returns `reloaded: false` with that message and applies nothing; `check-config` exits 1 listing each as refused; `model_copy` raises naming the new path |
| 23 | fused-memory (ξ) | load and `reload_config` a config with `reconciliation.sonnet_model: opus`; `get_status` on the live config | fixture config | refused naming `roles.recon_stage_standard.model`; the reload refuses with the same message; `get_status` carries no deprecated list |
| 24 | shared (π) | `canonical_model('fable')`, `('claude-fable-5-1')`, `('sonnet')`; `scope_for('fable', UsageCapConfig())`, `scope_for('sonnet', UsageCapConfig())` | default config | `'claude-fable-5-1'` twice and `'sonnet'`; `'claude-fable-5-1'` and `None`; `UsageCapConfig().scoped_cap_models == [FABLE_MODEL]` |

## Design invariant (ι performs the four-site lockstep edit)

Minted by ι: **`config-default-one-home`** (R22 item 5). The slug is the stable id. At `ce6673c7e1` the last
minted is still INV-14, and R51 reserves INV-15/16/17 for S6 9140, S10 9242 and S7 9154. The six wave-3
minting tasks (S1 ω, S1's eval-hermeticity task, S2 ι, this ι, S17's `one-decision-partition-per-area` and
S19 ε) take INV-18 to INV-23 in landing order, with `one-decision-partition-per-area` last as INV-23 (lead
ruling after the wave-3 fold-verifiers), so ι takes one of INV-18 to INV-22. The lead wires the
invariant-file chain at decompose and corrects the number there if the order changes. ι depends on 9154
because both edit the same three invariant files (a non-test file edge), not to wait for a number. Every
check anchors on the slug, never the number.

Sites (`scripts/tests/test_design_invariants_consistency.py::PINNED_SITES`):
`docs/legibility/design-invariants.md` (the section, the header dating paragraph, the `## Fixtures`
date list); `docs/legibility/design-invariants-fixtures.md` (`INV-N-PRD`, `INV-N-CODE`, and a dated
addendum walk with its rehearsal rows and the cumulative match count); `skills/prd/references/gates.md`
(the family-inventory row and the `inv-trigger-shapes` span). `CONTRIBUTING.md` gets nothing, because
it is pinned as an absence. The partial heuristic mapping in `agents/code_quality.md` is not edited.
Draft text for the normative section:

> **Rule**: A configuration default has exactly one home: the pydantic `Field` default (or
> `default_factory`) on the config model. No `orchestrator/src/orchestrator/defaults.yaml` leaf restates
> or overrides a pydantic field default except a `${VAR}` interpolation. A field whose default names a
> single shared constant is one home; a second literal of that value anywhere is not.
>
> **Checkable design question(s)**: Does this change add or change a config default? Is it stated only
> as the field default (or one shared constant the field default names), or does a yaml leaf state it
> too? Review guidance beyond the check: a module constant, a dashboard constant or a consumer fallback
> that repeats a default's value is the same defect, and a parity test between two spellings of a default
> is not compliance.
>
> **Evidence**: at `6551b75aba`, of 217 `defaults.yaml` leaves, 194 restated their field default and 23
> disagreed, so a bare submodel and a loaded config ran different values (`fk-2a67692ff456`, run
> `review-all-dark_factory-20261008`).
>
> **House pattern**: `orchestrator/tests/test_config_default_one_home.py` (C7).
>
> **Family boundary**: INV-9 `one-fact-one-home` governs world-facts written to records; this governs
> code configuration defaults. INV-5 `no-lockstep-duplication` accepts a drift test as a house pattern;
> this rejects it as enforcement.

Trigger shape for gates.md: *a config default stated anywhere but its pydantic field or the one shared
constant that field names (a defaults.yaml leaf that is not a `${VAR}` interpolation, a second literal)
(`config-default-one-home`)?*

Sites that the review guidance names and this plan leaves standing, each a deliberate exception or another
plan's: `scripts/orchestrator-watchdog.py::FLEET_LEASE_MAX_AGE_SECS` and `ORCH_RESTART_MIN_INTERVAL_SECS`
(stdlib-only mirrors the script keeps because it cannot import the orchestrator; drift-tested; S24 owns
the script) and `agents/invoke.py::invoke_agent`'s `startup_grace_secs=120.0` parameter default (review
guidance for the leaf that next edits that signature, which reads it from `agent_liveness`). α removes the
dashboard's two.

## Cross-PRD relationship

"X → Y" means X depends on Y (Y lands first). The order rule (OQ-1, ruled by the lead): on every non-test file this plan
shares with a wave-2 plan, the first S16 leaf to declare the file depends on that file's wave-2 tail
(R54's per-file order, `digests/wave2-order.json`, extended by the live store's per-file sinks for files
only one wave-2 stream edits). Later S16 leaves inherit the edge through their own chain. A consequence
measured on 2026-11-11: `config.py`'s tail is S8 θ 9229, which waits on 5040 like 21 of S8's 22 tasks, so
the orchestrator chain starts after 5040. The shared-only tasks and the fused-memory reload and provider
tasks (β0, γ0, β2, ε1) start without 5040; every task that needs the moved resolver (from ε2a on) and
every orchestrator task waits on it. (Measured: none of β2's or ε1's external dependencies — 9194, 9280,
9254, 9252, 8143, 8452, S15 ε1's and S19 δ's ancestors, S12 α's — reaches 5040; re-measured at
`ce6673c7e1`, when 9229's ancestry also reaches 8528.) Test files take no edge, inside the plan or across
plans (lead ruling after the wave-3 fold-verifiers, restating R54): an edge that would rest only on a
shared test file is not wired, and the scheduler's file locks serialise it.

| Other PRD / stream | Direction | Seam mechanism | Owner | Status / edge |
|---|---|---|---|---|
| S25 scripts instruments, C4 `shared.route_rejection` (task 6766) | consumes | `RouteRejection` on the shared resolver's decision | S25 | δ → 6766. 6766's absent-check on `routing.py` stays green after the move |
| S20 CLI invocation and usage gate (wave 3, folded and verified) | produces (to) and consumes | C1 `Backend` for `shared/invocation_outcome.py` (the pi arm), `shared/cli_invoke.py` (two comparisons) and `ACCOUNT_POOL_BACKENDS`; C1 now carries `ZCODE` and `MIMO` (tasks 5399, 7993), both outside the pool. S20 C6 `INVOCATION_COST_FLOOR_USD` for C5's budget check; S20 β moves `is_zero_output_timeout` to `shared/cli_result.py`; S20 ε moves `scope_for` to `shared/account_state.py` and S20 ι and ν edit that file too; π edits `scope_for`'s body (C9) | S16 owns C1 and C9; S20 owns C6 and converts its own sites | S20 ν → S16 γ0 (held on `S16/C1`, released by S16's decompose); δ → S20 δ; ζ6a → S20 β (ζ6a no longer declares `shared/cli_invoke.py`, as S20 asked); π → S20 ι (which follows S20 ε; S20 ν and π share `account_state.py` and the lead orders them at decompose). C1–C8 keep their numbers and every S20-facing label (γ0, δ, ζ6a) is unchanged; C9 and π are new. Xref returned |
| S24 fleet watchdog and redeploy (wave 3, draft) | same-file + sequencing | S24 γ deletes `orchestrator_restart_lease_max_age_secs` (its bound moves to `shared/fleet_coordination.py`); S24 ζ (the watched-path task) re-defaults `orchestrator_restart_watch_prefixes` from `WATCH_CLOSURE` and adds the restart-only `fused_memory_restart_min_interval_secs`, whose default is S24's `FM_ON_MERGE_MIN_INTERVAL_SECS` in `shared/fleet_coordination.py` (14400; S24 θ flips it to 3600 and waits on operator gate 9521, which puts task 9493's stop fix in effect; θ's one-line `OPERATIONS.md` edit is exempt from the same-file chain, so θ does not wait on S24 ω); both edit `config.py`, `harness.py`, `service_restart.py`, `scripts/orchestrator-watchdog.py` and two watchdog tests; ζ1 moves the restart fields S24 leaves, the new interval included, into `service_restart[fused_memory]` | each owns its edit; S24 owns the interval's value | ζ1 → S24 γ, S24 ζ (S16 moves what S24 left; `watch_prefixes`' and the interval's field defaults keep naming S24's constants in `shared/fleet_coordination.py`). No other S16 leaf takes an edge onto S24: α creates `shared/config_defaults.py` alone, and S24 never writes it (lead ruling after the wave-3 fold-verifiers). The author pass's reverse ask ("S24's spec leaf → S16 ζ1") is withdrawn. Xref returned |
| S17 scheduler seams (wave 3, draft) | same-file + sequencing (OQ-11, ruled by the lead) | S17 ζ deletes `OrchestratorConfig.stranded_blocked_redispatch_enabled`; S17's chain rewrites the scheduler phases ζ3 rewrites and moves scheduler readers into new modules (`backfill_admission.py`, `warm_base_watchdog.py`, `dispatch_ledger.py`, `tick_phases.py`, `delivered_check_gate.py`). Under S17's rulings task 4082 adds a hot-reloadable re-pend loop cap beside `stranded_reconcile_interval_secs`, and the phase-time alarm may add a threshold | each owns its edit | ζ2 → S17 ζ (which follows 4082); ζ3 → S17 κ1 (S17's last leaf on `scheduler.py`). One direction only: S17 writes its leaves against today's flat fields and carries no edge onto S16; S16 ζ2 and ζ3 rewrite them after S17 κ1, and their reader scans at filing pick up 4082's loop cap and any alarm threshold. Xref returned |
| S15 fused-memory MCP server boundary (wave 3, draft) | same-file | S15 ε1 moves the `reload_config` tool into `server/tool_families/ops.py` and edits `tests/test_config_reload.py` | each owns its edit | β2 → S15 ε1; β2's two checks are scoped to `fused-memory/src/fused_memory/server/`. S15 β's prerequisite 4867 is done (2026-11-13), so it no longer holds β2. Xref returned |
| S19 package layering (wave 3, draft) | same-file | S19 δ (72 files) edits fused-memory `config/reload.py`, `config/schema.py`, `server/tools.py`, `write_triage_judge.py`, `memory_service.py`, `scripts/exception_register.py`; S19 κ2 edits `dashboard/data/orchestrator.py`; S19 θ2 drops cockpit's orchestrator dependency (consistent with OQ-4) | each owns its edit | β2 → S19 δ; ε2b → S19 δ (also through β2); α → S19 κ2. Xref returned |
| S12 reconciliation run lifecycle (wave 3, draft) | same-file | S12 α edits `reconciliation/judge.py` and `tests/test_judge.py`; S12 β1 adds the run driver `drive` that ε2a's `_select_tier` rewrite rebases onto; S12 δ's prompt-and-skill sweep edits `CLAUDE.md` §"Write-Tagging Convention", which κ's prose sweep also edits (other lines) | each owns its edit | ε1 → S12 α; ε2a → S12 β1 (S12's draft: β1 never depends on ε2a); κ → S12 δ on `CLAUDE.md`, by label, wired at decompose (lead ruling; already implied through β1 → β2 → S15 ε1 → S12 δ, so no new wait and no cycle). Xref returned |
| S22 dashboard readers (wave 3, draft) | produces (to) | C8 for the dashboard's orchestrator defaults; S22 λ edits `dashboard/data/orchestrator.py` | S16 owns C8; S22 adds the escalation port to it | S22 λ → S16 α (S22's edge); λ's default sourcing follows OQ-4 (S22/OQ-5 is its duplicate). λ also makes a one-line `EscalationConfig.port` default edit in `orchestrator/src/orchestrator/config.py`; its place on this plan's `config.py` chain is set by the lead at decompose. Xref returned |
| S10 task-write contract | same-file | γ2 derives `KNOWN_ROLE_NAMES` in `shared/task_metadata.py` (values unchanged); ε2b edits `middleware/task_curator.py`; η edits `middleware/task_interceptor.py` | S10 owns the wire contract | γ2 → 9188 (S3 ε1, the file's tail); ε2b → 9204 (S10 δ); β2 → 9280 (S10 μ2, `server/tools.py`/`server/main.py`), 9254 (S10 λ, `reconciliation/harness.py`); η → 9282 (S10 ν, `task_interceptor.py`); ζ1 → 9251 (S10 κ2, `orphaned_recon_escalation_sweep.py`) |
| S26 memory service | same-file + sequencing | λ 9194 adds `identity_lock` to fm `schema.py`/`reload.py`/`config.yaml` and folds the mint lock timeout; η 9174 re-signs `resolve_topic_anchor_enabled`; τ 9245 and υ 9247 move the storm methods; μ 9203 is `graphiti_client.py`'s and `mem0_client.py`'s tail; φ 9252 is `memory_service.py`'s | S26 | β2 → 9194, 9252 (β2 carries λ's new tier entries onto markers); ε2b → 9203; η → 9174, 9245, 9247 |
| S8 harness composition root | same-file | `config.py`/`defaults.yaml`/`dark-factory-orchestrator.yaml`/`steward.py` tail θ 9229; `harness.py` tail ω 9236; `invoke.py` tail κ2 9233; `workflow.py`/`dry_run_unblock.py` tail κ1 9232; `scheduler.py` tail μ1 9234; `cli.py`/`evals/*.py`/`agents/briefing.py` tail μ2 9235 | S8 | α → 9229; β1 → 9236; γ1 → 9233; γ2 → 9232, 9235; ζ3 → 9234 |
| S4 verify run context | same-file | `verify.py`/`verify_plan.py`/`verify_cmd.py` tail ε 9160; `verify_runner.py`/`drift.py` tail ρ 9166; `task_verify_run.py`/`task_verify_dispatch.py`/`review_checkpoint.py` tail μ0 9150; `gates.py`/`history_leg.py` tail ν 9153; `task_verify_spec.py`/`verify_spec_shared.py` tail η 9135; `scripts/verify_budget_census.py` tail λ 9146. R28's `VerifyHost` reshapes host policy | S4 | α → 9146; γ2 → 9150; ζ2 → 9160; ζ6a → 9153; ζ4b → 9166, 9135 (ζ5b adopts whatever host-policy shape R28 landed) |
| S3 escalation server | same-file | `merge_lane/worker.py`, `cli_stage_runner.py` and `shared/task_metadata.py` tail ε1 9188; `OPERATIONS.md` tail μ 9210; `escalation/authority.py` tail κ 9173 | S3 | γ2 → 9188; ε2a → 9188; ζ4b → 9188; β1 → 9210; ζ1 → 9173 |
| S5 GitOps | same-file | `service_restart.py` (ρ0 9272, after ε 9263) | S5 | ζ1 → 9272 |
| S9 workflow spine | same-file | `routing_dispatch.py` tail ζ2 9202; `exit_contract.py` tail ν 9284 | S9 | γ2 → 9202; ζ6b → 9284 |
| S6 lane lifecycle | same-file + check | 9140 is first on `config.py` (R54). υ (R58, not yet filed) checks `lifecycle_refusal_policy: hold_lane` in `dark-factory-orchestrator.yaml` | S6 | none beyond the tails; κ keeps that spelling or the lead files υ against the new path. Xref returned |
| S7 gates (R51) and the wave-3 invariant leaves | same files | the three invariant files (`docs/legibility/design-invariants.md`, `docs/legibility/design-invariants-fixtures.md`, `skills/prd/references/gates.md`) | lead | ι → 9154 (S7 ζ, INV-17), a non-test file edge, not a wait for a number. The six wave-3 minting tasks take INV-18 to INV-23 in landing order, `one-decision-partition-per-area` (S17) last as INV-23; the lead wires the invariant-file chain among them at decompose |
| S1 one escalation filing entry point (wave 3) | shares files | S1's corpus-sandbox task stops `evals/runner.py::build_eval_orch_config` from setting `SandboxConfig(enabled=False)`; S1's eval-hermeticity task edits `evals/profile.py` and `workflow.py`. α keeps `sandbox.enabled` False as the loaded default (D2), so S1's corpus cells must take the sandbox block from the project's loaded config, not from a bare `SandboxConfig()` | each owns its edit | `evals/runner.py` (γ2, δ, ζ6a), `evals/profile.py` (β1, ζ6b) and `workflow.py`: the lead orders them at decompose under the per-file rule. Xref returned |
| S2 typed escalation record facts (wave 3) | shared shape + mechanism | S2's identity-header deploy to other projects' `.mcp.json` (S2 ν) uses the same operator-sweep gate shape as σ and the same pre-authorised-spawn mechanism, χ's `metadata.x_preauthorised_spawn` exception in the escalation-watcher skill; the two gates are separate and share no file | S16 owns χ (lead ruling after the wave-3 fold-verifiers: the mechanism stays in S16); S2 owns ν | S2 ν → S16 χ (χ has no prerequisite inside this plan, the lead's condition for this edge; wired by the lead at the join). S2 ν carries the metadata key |
| `plans/merge-lane-policy-prd.md` (8526–8528; 8525 done) | parking | `merge_lane/worker.py` | that PRD | ζ4b, ζ5a, ζ5b → 8528 (R55 parking; Leo's 2026-11-10 chain protection). 7720 (deep merge-ahead) is done and leaves this row |
| `plans/routing-experiment-substrate-prd.md` (8373 high, 8375, 8390) | precedes α and δ | 8373 adds `SplitMatch` to `RuleMatch` and a split condition to `_rule_matches`; 8375 sets the experiment rule table in `defaults.yaml` and the project yaml | that PRD | α → 8375 (α absorbs 8375's table as the field default; 8375 → 8373); δ → 8373. δ moves `SplitMatch` with the rule models; 8373's check needs the clerk's path amendment (§Delivered checks on moved paths) |
| task 5366 (high; dispatch-path gates fail open) | precedes β1 and δ | the ceiling check on `_model_rejection_reason` / `spend_by_model`; it also edits `harness.py`, `invoke.py`, `routing.py`, `routing_dispatch.py`, `workflow.py` | 5366 | β1 → 5366 (β1 is the first S16 leaf declaring one of its files); δ keeps its fail-closed semantics |
| task 3792 (flake ledger θ) | precedes β1 | adds a `_submodel_leaf_paths('flake_ledger')` entry and a check on it | 3792 | β1 → 3792; its check is amended (§Delivered checks on moved paths) |
| tasks 8143, 8452 | precede β2 | add fused-memory reload tier entries to `config/reload.py` and checks on them | those PRDs | β2 → 8143, 8452; their checks are amended |
| task 5040 (verify-dispatch extraction) | transitive | 90 declared files incl. `worker.py`, `verify*.py` | that PRD | every orchestrator S16 leaf waits on it through the tails above; no direct edge |
| task 5407 (pending, low; the stale `claude-fable-5` scoped-cap literal) | absorbed | `scoped_cap_models` default and its tests | S16 π | cancel with `x_acceptance_reason` naming π (lead action) |
| task 8253 (pending, low; rescoped to the return-brief half by Leo's 2026-11-12 ruling) | none | no longer edits `usage_gate.py`, `config_models.py` or `account_pool.py` | 8253 | no shared file with π; no edge |

## Delivered checks on moved paths (for the decompose `moves` and the clerk scan)

Clerk instruction: at each leaf's filing, scan `metadata.delivered_checks` of every non-terminal task (the
forensic read of `tasks.db`), not only the committed sidecars, for patterns this leaf's moves turn red, and
amend each to a form that passes on both sides of the move (an alternation of the old and new spelling, or
the old and new path in `paths`; `git grep` treats a deleted path as no match, not an error). The scan on
2026-11-11 found these (re-read 2026-11-13: every pattern unchanged), each carried as a `lead_actions` entry. At β1's and β2's filing the clerk also scans
for checks that pin a GREEN field's annotation spelling (`: <type> = Field`), since markers rewrite
annotations without moving anything:

- **9229** (S8 θ, pending; dependent 9166, an ancestor of 9188, which γ2 needs): drop
  `default-cap-25-in-defaults-yaml` (`^steward_lifetime_budget: 25\.0` on `defaults.yaml`, deleted by α)
  and widen `default-cap-25-in-config` to
  `lifetime_budget: (Annotated\[)?float(, Reload\.GREEN\])? = Field\(default=25\.0` over
  `orchestrator/src/orchestrator/` (it then holds after β1 marks the field GREEN and after ζ6b moves it into
  `steward`).
  Filed before α; otherwise 9166 → 9188 → γ2 is held by S16's own α.
- **3792** (flake ledger θ, pending; dependent 3795): widen `_submodel_leaf_paths\(['"]flake_ledger['"]` to
  `_submodel_leaf_paths\(['"]flake_ledger['"]|flake_ledger: Annotated\[FlakeLedgerConfig, Reload\.GREEN\]`.
- **8143** (pending; dependents 8144, 8145, 8146, 8453, 9180) and **8452** (pending; dependents 8453, 8455,
  8440): their `cancel_children\.enforce` and `curator\.pack_sweep\.` checks on fused-memory
  `config/reload.py` become `<old pattern>|<the marker β2 writes>` over `config/reload.py` and
  `config/schema.py`; β2 states the marker spelling.
- **4651** (pending, an ancestor of 9188): its new knob defaults go on the `config.py` fields, never in
  `defaults.yaml`; at ζ2's filing its `landed_reconcile[a-z_]*_(interval_secs|enabled|max[a-z_]*)` check
  widens to cover the nested `recovery_sweeps` class over `orchestrator/src/orchestrator/`.
- **8373** (pending; dependents 8374–8377): add `shared/src/shared/route_policy.py` to the `class SplitMatch`
  check's paths at δ's filing.
- **3587** (pending; dependent 9228) `milestone_gate` and **8528** `depth_table`, both scoped to
  `config.py`: OQ-9 was ruled option 2 (lead), so the moved defaults land in `config_sections/`, and the
  clerk amends both at ζ1's and ζ5a's filing to paths `orchestrator/src/orchestrator/`.
- **Done tasks** whose checks name moved spellings: 5237, 5806, 6148, 6181 (fused-memory
  `config/reload.py`, deleted by β2), 5629 (`^verify_admission_order:` in `dark-factory-orchestrator.yaml`,
  nested by κ) and 2858 (`routing.py` `model-capacity-exhausted`, δ). The clerk amends one only if a
  dependent is still pending. κ is filed only after the yaml-anchored amendments.
- **Pending `defaults.yaml` editors.** 47 non-terminal tasks declare `orchestrator/src/orchestrator/defaults.yaml`
  in `metadata.files` (measured 2026-11-13; 46 on 2026-11-11). Any of them that lands after α and adds a literal leaf fails
  C7. 6754, 5377 and 4651 get explicit amendments; the rest are a program-level instruction (xref): at
  α's filing, the clerk amends each still-pending one to "edit the field default in `config.py`, never
  `defaults.yaml`".

## Decomposition plan

Twenty-eight leaves; none `high`/`critical`, so none is a dependency of gate 6698. Every leaf carries
`x_finding_run = "review-all-dark_factory-20261008"`, `source = "review-all"` and file-level
`metadata.files`; every leaf but the gate σ, the watcher rule χ and π carries `metadata.x_finding_key` (list). Intra-plan edges
(X → Y: X depends on Y): β1 → α, β0, β2; γ1 → β1, γ0; γ2 → γ1; δ → γ2; ζ1 → δ; ζ2 → ζ1; ζ3 → ζ2;
ζ4a → ζ3; ζ6a → ζ4a; ζ6b → ζ6a; ζ4b → ζ6b; ζ5a → ζ4b; ζ5b → ζ5a; κ → ζ5b; σ → ζ5b, κ, χ; ξ → σ, η;
π → κ; ι → α; β2 → β0; ε1 → β2, γ0; ε2a → ε1, δ; ε2b → ε2a; η → ε2b; ω → ξ, ι. β0, γ0 and χ have none.
Out-of-batch edges are the tails in §Cross-PRD relationship; each leaf's `Deps:` line lists the ones it
carries directly. No intra-plan edge rests only on a test file. Task 9490 makes the task store refuse a
cycle-closing edge, so the decompose wiring needs no cycle pre-pass of its own (one may stay as an early
check).

Each ζ leaf is a mechanical codemod of one subsystem's field group: it moves the fields with their markers
and legacy spellings, rewrites every reader, rewrites the tests that construct or read the moved fields,
and extends boundary row 18's golden map. Each declares every file the reader scan found at `ba17934861`,
re-run at `ce6673c7e1` by the ruling fold
(attribute, string key, keyword argument, yaml key), so a leaf's `Files:` line is long. The overlay's file
target cannot be met by splitting further: single fields reach 44–61 test files on their own (§Fold
measurements), and each test site is a one- or two-line rewrite of a hand-built config double. Splitting a
field group would instead leave both spellings readable in code, the stitched intermediate heuristic 13
forbids. The completion checks of ζ3, ζ4b, ζ5a, ζ5b, ζ6a and ζ6b are reader-side (no production reader
names a moved field at the config root, attribute or `getattr` string), paired with a present check on the
new root declaration, because the submodel keeps the sub-prefixes a class-body grep would match; ζ1, ζ2
and ζ4a keep class-body checks because their submodel names absorb the whole prefix.

**β0 — Shared config-field declarations (reload tier, legacy spelling, role-table pivot) and the generic hot-reload engine** (`task_kind=normal`, priority medium, est. ~800 LOC)
- Files (4): `shared/src/shared/config_fields.py` (new), `shared/src/shared/config_reload.py` (new), `shared/tests/test_config_fields.py` (new), `shared/tests/test_config_reload.py` (new).
- Keys: `fk-efae65b1f9af`, `fk-dae2a050b5d3` (the tier-on-field half).
- Deps: none.
- Work: C3 and C4. The engine is written from fused-memory's port (annotation-driven descent, `_ABSENT`, path union), generalised over the model class, with `Mapping[StrEnum, BaseModel]` descent added for role tables. `TableLegacy` and its `knob_major` form land here with the scalar `LegacySpelling`, so the first leaf that needs a table pivot (γ2) finds it. Boundary rows 1, 2, 4 and 4b run against fixture models, 4b against a fixture model with a `dict[StrEnum, BaseModel]` table.
- Deletes: nothing yet (β1 and β2 delete the two engine copies).
- Observable signal: consumed by β1, β2 and γ2. The intermediate's own signal is boundary rows 1, 2, 4 and 4b green in CI.
- G7: INV-5: β0 adds the third copy only transiently, and β1 and β2 delete the other two, so the batch ends with one. INV-2: every error names the path and the values.

**γ0 — Shared role and backend vocabulary: Backend, RoleName, config_role, RoleSettings** (`task_kind=normal`, priority medium, est. ~500 LOC)
- Files (4): `shared/src/shared/backends.py` (new), `shared/src/shared/roles.py` (new), `shared/tests/test_backends.py` (new), `shared/tests/test_roles.py` (new).
- Keys: `fk-ba569525f6e0`, `fk-5805ae6f97f1` (the vocabulary half).
- Deps: none.
- Work: C1 and C2, including `REVIEWER_VARIANTS` and `DISPATCHABLE_ROLES`. C1 carries all six CLI backends `invoke_agent` dispatches at `ce6673c7e1` (zcode and mimo included). `RoleSettings` and `merge_role_table` land here so γ2 and ε2a import one definition. Tactical 3 (the `effort` type) is decided here.
- Deletes: nothing.
- Observable signal: boundary row 7. Consumers: γ1, γ2, ε1, ε2a, and S20 ν (held on `S16/C1`).
- G7: INV-11 (unknown values raise loudly); INV-2.

**α — Config defaults have one home: field defaults are authoritative, defaults.yaml keeps only environment-layered leaves** (`task_kind=normal`, priority medium, est. ~1,400 LOC, mostly yaml deletion)
- Files (34): `orchestrator/src/orchestrator/config.py`, `orchestrator/src/orchestrator/defaults.yaml`, `shared/src/shared/config_defaults.py` (new), `dashboard/src/dashboard/data/orchestrator.py`, `dashboard/src/dashboard/data/scheduler.py`, `scripts/verify_budget_census.py`, `orchestrator/tests/test_config_default_one_home.py` (new), `orchestrator/tests/fixtures/config_default_one_home/` (new, three seeded yamls), `orchestrator/tests/_packaged_defaults.py`, `orchestrator/tests/test_config.py`, `orchestrator/tests/test_chronic_flake.py`, `orchestrator/tests/test_config_merge_deep.py`, `orchestrator/tests/test_dry_run_unblock.py`, `orchestrator/tests/test_interactive_worktree.py`, `orchestrator/tests/test_merge_queue_orphan_reaper.py`, `orchestrator/tests/test_roles_wait_pattern.py`, `orchestrator/tests/test_verify_clock_stop.py`, `orchestrator/tests/test_zero_progress_requeue.py`, `orchestrator/tests/test_config_reload_integration_gate.py`, `orchestrator/tests/test_sandbox_fleet_census.py`, `orchestrator/tests/test_workflow_helpers.py`, `orchestrator/tests/test_merge_queue_multihost_wiring.py`, `orchestrator/tests/test_harness_park_stop.py`, `orchestrator/tests/test_eval_boundary_suite.py`, `orchestrator/tests/test_landlock.py`, `orchestrator/tests/test_review_checkpoint_pre_test.py`, `orchestrator/tests/test_routing.py`, `orchestrator/tests/test_routing_config.py`, `orchestrator/tests/test_routing_resolver.py`, `tests/scripts/test_orchestrator_config_duplicate_keys.py`, `dashboard/tests/test_orchestrator.py`, `dashboard/tests/test_scheduler_page.py`, `dashboard/tests/test_scheduler_state_access.py`, `tests/scripts/test_verify_budget_census_regimes.py`.
- Keys: `fk-2a67692ff456`.
- Deps: 9229 (S8 θ, the tail of `config.py` and `defaults.yaml`); 8375 (the experiment-1 rule table, which brings 8373; α absorbs 8375's table as the field default); 9146 (S4 λ, the tail of `scripts/verify_budget_census.py`); S19 κ2 (`dashboard/data/orchestrator.py`). No edge onto S24 (lead ruling: S24 homes its restart defaults in `shared/fleet_coordination.py`).
- Work: D2, D3, C7, C8 (OQ-4, ruled by the lead). α creates `shared/config_defaults.py`. Each of the 23 disagreements takes its survivor as the field default (D2's drift rule applies). Every restating leaf is deleted, and the price-table lockstep test goes with its restatement. `orchestrator/tests/_packaged_defaults.py::load_packaged_defaults` is repointed to a public field-default reader over `OrchestratorConfig.model_fields`, so the tests that read packaged values keep one seam (Tests stance) and stop `KeyError`ing on deleted leaves. The listed tests parse `defaults.yaml` (measured at `ba17934861` and re-run at `ce6673c7e1`: a `safe_load` or packaged-defaults read in the same file); each is moved onto that seam or onto the field default. `scripts/verify_budget_census.py`'s `_MERGE_BUDGET_SOURCES` stops reading `merge_verify_cold_command_timeout_secs` raw from `defaults.yaml` and reads the field default. The dashboard imports `DEFAULT_MAX_CONCURRENT_TASKS` and `DEFAULT_LOCK_DEPTH` from C8, and the `max_concurrent_tasks`/`lock_depth` field defaults name the same constants. Bare-construction tests that asserted an old submodel default are updated to the survivor (the listed `SandboxConfig()`/`ReviewConfig()`/knob-major `()` sites).
- Deletes: the 203 restating leaves (`prices` included) and their comment prose (operator facts move to descriptions); `_ORCHESTRATOR_DEFAULT_MAX_CONCURRENT_TASKS` and the dashboard drift test; `dashboard/data/scheduler.py::_DEFAULT_LOCK_DEPTH` (2, against the survivor 4); `test_default_price_table_matches_defaults_yaml`.
- Observable signal: the §Goal one-liner prints `24 24`. The guard is green on the live tree and flags all three seeded fixtures (row 16). `orchestrator check-config` on dark-factory's yaml reports the same resolved values as before α, with the `model_dump()` diff empty (row 18 shape).
- G7: the new invariant holds by construction; INV-10 (the guard runs on fixtures, not prose); INV-13 (the dashboard reads the value the orchestrator actually uses).

**ι — Promote the config-default-one-home design invariant through the four-site lockstep edit** (`task_kind=normal`, priority medium, est. ~200 LOC of docs)
- Files (3): `docs/legibility/design-invariants.md`, `docs/legibility/design-invariants-fixtures.md`, `skills/prd/references/gates.md`.
- Keys: `fk-2a67692ff456`.
- Deps: α (the house pattern exists); 9154 (S7 ζ `merge-gate-unmeasured-typed`, INV-17, the last of R51's wave-2 invariant chain; same three invariant files); the lead orders it against the other wave-3 invariant leaves (S1 ω, S1's eval-hermeticity task, S2 ι, S17's `one-decision-partition-per-area`, S19 ε) at decompose. ι takes one of INV-18 to INV-22; checks anchor on the slug.
- Work: §Design invariant.
- Deletes: nothing.
- Observable signal: `uv run --project shared pytest scripts/tests/test_design_invariants_consistency.py` passes with the new slug in every pinned site.
- G7: INV-10, the fixture walk in the addendum.

**β2 — Fused-memory declares reload tiers and legacy spellings on its fields; its engine copy config/reload.py is deleted** (`task_kind=normal`, priority low, est. ~900 LOC)
- Files (19): `fused-memory/src/fused_memory/config/schema.py`, `fused-memory/src/fused_memory/config/reload.py` (deleted), `fused-memory/src/fused_memory/server/tools.py`, `fused-memory/src/fused_memory/server/tool_families/ops.py`, `fused-memory/src/fused_memory/server/main.py`, `fused-memory/src/fused_memory/reconciliation/harness.py`, `fused-memory/src/fused_memory/server/write_triage.py`, `fused-memory/src/fused_memory/server/write_triage_judge.py`, `fused-memory/src/fused_memory/services/memory_service.py`, `fused-memory/src/fused_memory/mcp_tools/scheduler_state.py`, `scripts/exception_register.py`, `fused-memory/tests/server/test_write_triage_attach_guard.py`, `fused-memory/tests/server/test_write_triage_judge.py`, `fused-memory/tests/test_config_reload.py`, `fused-memory/tests/test_config_schema.py`, `fused-memory/tests/test_config_write_triage_project_calibrations.py`, `fused-memory/tests/test_link_heal_config.py`, `fused-memory/tests/test_overreach_shadow_wiring.py`, `fused-memory/tests/test_task_interceptor.py`.
- Keys: `fk-efae65b1f9af`.
- Deps: β0; 9194 (S26 λ, the tail of fused-memory `config/schema.py`, `config/reload.py` and `config/config.yaml`: β2 carries λ's `identity_lock` tier onto a marker); 9280 (S10 μ2, `server/tools.py`, `server/main.py`); 9254 (S10 λ, `reconciliation/harness.py`); 9252 (S26 φ, `services/memory_service.py`); 8143 and 8452 (their `cancel_children.enforce` and `curator.pack_sweep.*` tier entries ride onto markers); S15 ε1 (moves the `reload_config` tool into `server/tool_families/ops.py` and edits `tests/test_config_reload.py`); S19 δ (edits `config/reload.py`, `config/schema.py`, `server/tools.py`, `memory_service.py`, the write-triage modules and `scripts/exception_register.py`).
- Work: Each entry of fused-memory's 259-line allowlist becomes a marker. A root `model_validator(mode='before')` on `FusedMemoryConfig` calls `pivot_legacy_spellings` (it has no legacy spelling to pivot until ε1, and it is the one place every later fused-memory move declares into). Fused-memory's `reload_config` tool calls C4 wherever S15 ε1 placed it under `server/`, and its result lists the `DeprecatedSpelling`s; the `get_status` tool result gains a `config.deprecated_spellings` list (fused-memory has no `check-config` and no census, so these two are its report surfaces). The `ListRef(…, 'RELOADABLE_FIELDS')` row for fused-memory in `scripts/exception_register.py` is retargeted to the derived constant. The tier prose at the sites that restate `RELOADABLE_FIELDS` (`schema.py` ×2, `reconciliation/harness.py`, `server/main.py`, `server/tools.py`, `server/write_triage.py`, `server/write_triage_judge.py`, `services/memory_service.py`) is cut to a pointer, and `mcp_tools/scheduler_state.py`'s restatement of the orchestrator tier goes. The listed tests import from `fused_memory.config.reload` and move to `shared.config_reload`.
- Deletes: `fused_memory/config/reload.py` (465 lines).
- Observable signal: `mcp__fused-memory__reload_config` against the running server with a `write_triage.judge_timeout_seconds` edit reports it `applied`. Boundary row 1 on the fused-memory shape.
- G7: INV-5 (the second engine copy goes); INV-12 (the register row is retargeted, not dropped).

**β1 — The orchestrator declares reload tiers and legacy spellings on its fields; its engine copy goes** (`task_kind=normal`, priority medium, est. ~1,300 LOC)
- Files (56): `orchestrator/src/orchestrator/config.py`, `orchestrator/src/orchestrator/harness.py`, `orchestrator/src/orchestrator/cli.py`, `orchestrator/src/orchestrator/evals/profile.py`, `scripts/exception_register.py`, `OPERATIONS.md`, `escalation/tests/test_reload_config_integration.py`, `orchestrator/tests/_offline_lane_red_path_fixtures.py`, `orchestrator/tests/_psi_admission_yaml.py`, `orchestrator/tests/_task_verify_integration_rig.py`, `orchestrator/tests/test_config.py`, `orchestrator/tests/test_config_deterministic.py`, `orchestrator/tests/test_config_disjoint_skip_reload.py`, `orchestrator/tests/test_config_implementer_brief_variant.py`, `orchestrator/tests/test_config_merge_deep.py`, `orchestrator/tests/test_config_merge_gate_history_pure.py`, `orchestrator/tests/test_config_plan_text_fp_reload.py`, `orchestrator/tests/test_config_pre_merge_reverify_reload.py`, `orchestrator/tests/test_config_psi_admission_reload.py`, `orchestrator/tests/test_config_reload_integration_gate.py`, `orchestrator/tests/test_config_scoring_reload.py`, `orchestrator/tests/test_config_shadow_eval.py`, `orchestrator/tests/test_config_task_verify_policy.py`, `orchestrator/tests/test_config_unknown_keys.py`, `orchestrator/tests/test_config_verify_admission_reload.py`, `orchestrator/tests/test_config_verify_cache.py`, `orchestrator/tests/test_config_verify_cgroup_cpu_weight.py`, `orchestrator/tests/test_config_verify_cgroup_memory.py`, `orchestrator/tests/test_config_verify_host_policy.py`, `orchestrator/tests/test_eval_profile.py`, `orchestrator/tests/test_eval_shadow_invocation.py`, `orchestrator/tests/test_harness_config_reload.py`, `orchestrator/tests/test_harness_config_unknown_keys.py`, `orchestrator/tests/test_harness_escalation_revalidation_sweep.py`, `orchestrator/tests/test_merge_queue_deep_dispatch.py`, `orchestrator/tests/test_merge_queue_deep_integration_gate.py`, `orchestrator/tests/test_merge_queue_deep_landing.py`, `orchestrator/tests/test_offline_lane.py`, `orchestrator/tests/test_offline_lane_policy_reload.py`, `orchestrator/tests/test_review_checkpoint_pre_test.py`, `orchestrator/tests/test_routing.py`, `orchestrator/tests/test_routing_integration_gate.py`, `orchestrator/tests/test_routing_reload.py`, `orchestrator/tests/test_scheduler_park_backfill.py`, `orchestrator/tests/test_scheduler_pin_reservations.py`, `orchestrator/tests/test_scheduler_tick_phases.py`, `orchestrator/tests/test_task_verify_dispatch.py`, `orchestrator/tests/test_transcript_archive_config.py`, `orchestrator/tests/test_transient_requeue_backoff.py`, `orchestrator/tests/test_verify_admission_pytest_n.py`, `orchestrator/tests/test_verify_scope_cpu_weight.py`, `orchestrator/tests/test_verify_scope_memory.py`, `scripts/tests/config_reload_script_fakes.py`, `scripts/tests/test_merge_deep_set_cap.py`, `scripts/tests/test_merge_pytest_n_ab_switch.py`, `orchestrator/tests/test_config_model_copy_pivot.py` (new).
- Keys: `fk-dae2a050b5d3`, `fk-efae65b1f9af` (closing).
- Deps: α, β0, β2 (same file `scripts/exception_register.py`: fused-memory retargets its register row first); 9236 (S8 ω, the tail of `harness.py`; later leaves inherit it); 9210 (S3 μ, the tail of `OPERATIONS.md`); 3792 (flake ledger θ adds a `_submodel_leaf_paths('flake_ledger')` entry that β1 converts to a marker); 5366 (it also edits `harness.py`, `routing.py`, `routing_dispatch.py`, `workflow.py` and `invoke.py`; β1 is the first S16 leaf to declare one of them).
- Work: D4, D6 (its translation phase). Every entry of today's 240-path allowlist (measured at `ce6673c7e1`) becomes a `Reload.GREEN` marker on its field, or on its submodel field for a whole-submodel group. `RELOADABLE_FIELDS = reloadable_paths(OrchestratorConfig)`. The root `mode='before'` validator calls `pivot_legacy_spellings`. `census_config_keys` reports declared legacy spellings as `deprecated` (a new census bucket beside unknown and ignored); `cli.py`'s `check-config` prints that bucket, and the reload report lists them in C4's `deprecated` key. `OrchestratorConfig.model_copy` is overridden so an `update=` key that names a declared legacy spelling is pivoted and an unknown key raises (boundary row 18b): today `OrchestratorConfig().model_copy(update={'no_such_knob': 7})` succeeds silently, so every later move would leave old-key test sites green against a dead attribute. The harness's `reload_config` calls C4, and `evals/profile.py` (which imports `_iter_leaves`) calls `shared.config_reload.iter_leaves`. The orchestrator's `ListRef(…, 'RELOADABLE_FIELDS')` row in `scripts/exception_register.py` is retargeted to the derived constant. `OPERATIONS.md` §"Config reload vs restart" points at the markers. The listed tests import the engine symbols this leaf deletes and move to `shared.config_reload`.
- Deletes: the 338-line `RELOADABLE_FIELDS` block (220 comment lines), `_submodel_leaf_paths`, `ConfigDiff`, `_iter_leaves`, `diff_config`, `_get_leaf`, `_set_leaf`, `apply_reload` from `config.py`.
- Observable signal: boundary rows 3, 5, 6 and 18b. `mcp__escalation__reload_config` on the running dark-factory orchestrator with an unchanged yaml returns `reloaded: true` with empty `applied`, and a `review.interval` edit lands in `applied`.
- G7: INV-12 (the register rows are retargeted, not dropped); INV-11 (a deprecated spelling is reported, never ignored; an unknown `model_copy` key raises); INV-4: the census reports a deprecation once per load and files nothing, so it needs no storm escape.

**γ1 — Orchestrator backend fields are Backend; invoke_agent dispatches through one total table** (`task_kind=normal`, priority medium, est. ~450 LOC)
- Files (5): `orchestrator/src/orchestrator/config.py`, `orchestrator/src/orchestrator/agents/invoke.py`, `orchestrator/tests/test_invoke.py`, `orchestrator/tests/test_config.py`, `orchestrator/tests/test_routing_reload.py`.
- Keys: `fk-ba569525f6e0`.
- Deps: β1, γ0; 9233 (S8 κ2, the tail of `invoke.py`).
- Work: the 13 backend fields are typed `Backend`. `_validate_models_in_allowlist` compares enum values. `invoke_agent` looks its invoker up in a `Mapping[Backend, Invoker]` over `AGENT_CLI_BACKENDS` (six members at `ce6673c7e1`: claude, codex, gemini, pi, zcode, mimo), asserted total at import. An SDK backend handed to `invoke_agent` raises naming the backend.
- Deletes: the `if/elif` chain and the dispatch-time `ValueError('Unknown backend')`.
- Observable signal: boundary row 8 against the running orchestrator's `reload_config`.
- G7: INV-11; INV-2.

**γ2 — One role-keyed RoleSettings table replaces the knob-major submodels, the unblock-auto role knobs and the watcher role knobs** (`task_kind=normal`, priority medium, est. ~1,600 LOC)
- Files (88): `orchestrator/src/orchestrator/config.py`, `orchestrator/src/orchestrator/routing.py`, `orchestrator/src/orchestrator/workflow.py`, `orchestrator/src/orchestrator/steward.py`, `orchestrator/src/orchestrator/harness.py`, `orchestrator/src/orchestrator/review_checkpoint.py`, `orchestrator/src/orchestrator/routing_dispatch.py`, `orchestrator/src/orchestrator/dry_run_unblock.py`, `orchestrator/src/orchestrator/evals/runner.py`, `orchestrator/src/orchestrator/evals/configs.py`, `orchestrator/src/orchestrator/evals/architect_eval.py`, `orchestrator/src/orchestrator/agents/invoke.py`, `shared/src/shared/task_metadata.py`, `escalation/tests/test_reload_config_integration.py`, `orchestrator/tests/_orch_helpers.py`, `orchestrator/tests/_review_loop_harness.py`, `orchestrator/tests/conftest.py`, `orchestrator/tests/test_b3_gate.py`, `orchestrator/tests/test_config.py`, `orchestrator/tests/test_config_reload_integration_gate.py`, `orchestrator/tests/test_conftest_helpers.py`, `orchestrator/tests/test_dry_run_unblock.py`, `orchestrator/tests/test_eval_architect.py`, `orchestrator/tests/test_eval_candidate_bundles.py`, `orchestrator/tests/test_eval_codex_pi_bundles.py`, `orchestrator/tests/test_eval_driver.py`, `orchestrator/tests/test_eval_profile.py`, `orchestrator/tests/test_eval_shadow_invocation.py`, `orchestrator/tests/test_harness_config_reload.py`, `orchestrator/tests/test_harness_module_tagging.py`, `orchestrator/tests/test_harness_watcher_supervisor.py`, `orchestrator/tests/test_inflight_verify_merge_lease.py`, `orchestrator/tests/test_info_l0_router_workflow.py`, `orchestrator/tests/test_invoke_role_config_resolution.py`, `orchestrator/tests/test_laptop_warm_verify_boundary.py`, `orchestrator/tests/test_mark_blocked_outcome_scoping.py`, `orchestrator/tests/test_merge_boundary_effective_module_configs.py`, `orchestrator/tests/test_merge_gates.py`, `orchestrator/tests/test_merge_gates_equivalence.py`, `orchestrator/tests/test_merge_gates_git_probes.py`, `orchestrator/tests/test_merge_queue.py`, `orchestrator/tests/test_merge_queue_concurrent_verify.py`, `orchestrator/tests/test_merge_queue_depth_telemetry.py`, `orchestrator/tests/test_merge_queue_disposition_wiring.py`, `orchestrator/tests/test_merge_queue_dry_run_unblock.py`, `orchestrator/tests/test_merge_queue_flake_recorder.py`, `orchestrator/tests/test_merge_queue_main_health.py`, `orchestrator/tests/test_merge_queue_main_health_deferred.py`, `orchestrator/tests/test_merge_queue_multihost_wiring.py`, `orchestrator/tests/test_merge_queue_verify_retry.py`, `orchestrator/tests/test_merge_queue_warm_cold_shadow.py`, `orchestrator/tests/test_merge_reverify_gate_record.py`, `orchestrator/tests/test_model_overrides_end_to_end.py`, `orchestrator/tests/test_multihost_verify_integration.py`, `orchestrator/tests/test_out_of_band_routing.py`, `orchestrator/tests/test_review_cap_exit.py`, `orchestrator/tests/test_review_checkpoint_cap.py`, `orchestrator/tests/test_review_checkpoint_degraded_outcomes.py`, `orchestrator/tests/test_review_checkpoint_pre_test.py`, `orchestrator/tests/test_review_checkpoint_working_tree.py`, `orchestrator/tests/test_routing.py`, `orchestrator/tests/test_routing_byte_equivalence.py`, `orchestrator/tests/test_routing_dispatch.py`, `orchestrator/tests/test_routing_integration_gate.py`, `orchestrator/tests/test_routing_one_step_bounds.py`, `orchestrator/tests/test_routing_resolver.py`, `orchestrator/tests/test_routing_theta_plan_shape.py`, `orchestrator/tests/test_scope_capacity_gate_integration.py`, `orchestrator/tests/test_simple_task_workflow.py`, `orchestrator/tests/test_steward.py`, `orchestrator/tests/test_verify_admission_call_sites.py`, `orchestrator/tests/test_verify_plan_integration.py`, `orchestrator/tests/test_verify_scope_spawn_failure.py`, `orchestrator/tests/test_workflow_e2e.py`, `orchestrator/tests/test_workflow_escalated_steward_stall.py`, `orchestrator/tests/test_workflow_invocation_end_truthful.py`, `orchestrator/tests/test_workflow_mark_blocked_l0_coverage.py`, `orchestrator/tests/test_workflow_merge_gating_strand.py`, `orchestrator/tests/test_workflow_merge_retry_resume_merge_entry.py`, `orchestrator/tests/test_workflow_metadata_files_reconcile.py`, `orchestrator/tests/test_workflow_prompt_builder.py`, `orchestrator/tests/test_workflow_review_persistence.py`, `orchestrator/tests/test_workflow_routing_decision.py`, `orchestrator/tests/test_workflow_status_on_resume.py`, `orchestrator/tests/test_role_table.py` (new), `orchestrator/tests/_info_l0_router_fixtures.py`, `orchestrator/tests/test_eval_mimo_code_bundles.py`, `orchestrator/tests/test_eval_zcode_bundle.py`.
- Keys: `fk-5805ae6f97f1`.
- Deps: γ1; 9232 (S8 κ1, the tail of `workflow.py` and `dry_run_unblock.py`); 9188 (S3 ε1, `shared/task_metadata.py`); 9150 (S4 μ0, `review_checkpoint.py`); 9202 (S9 ζ2, `routing_dispatch.py`); 9235 (S8 μ2, `evals/runner.py` and `evals/architect_eval.py`). `steward.py`'s tail 9229 comes through α.
- Work: D7, D8. Every per-role read goes through `config.roles[config_role(name)]`, with no `getattr`. `evals/runner.py::build_eval_orch_config` builds role rows. The `roles` field carries one `TableLegacy.knob_major` per old block (`models`→`model`, `budgets`→`budget_usd`, `max_turns`→`max_turns`, `effort`→`effort`, `timeouts`→`timeout_secs`, `backends`→`backend`) and one `TableLegacy` row map for `unblock_auto.<knob>`, the five `watcher_*` role knobs and the `simple_task_*` scalars, so every operated yaml's old spelling pivots into the table (D6). `startup_grace_secs`/`working_idle_secs` move to `agent_liveness` with scalar `LegacySpelling`s. `KNOWN_ROLE_NAMES = {r.value for r in DISPATCHABLE_ROLES} | REVIEWER_VARIANTS`, and the orchestrator parity test asserts `set(ROLES) ⊆ KNOWN_ROLE_NAMES`. The listed tests construct the knob-major classes, read `config.<block>.<role>`, or set the moved role knobs (measured at `ba17934861`, re-run at `ce6673c7e1`). γ2 re-runs boundary row 6 against its base list and names each tier change in its commit; the five `watcher_*` role knobs are restart-only today, so the watcher row either stays in `RestartOnlyRows` or its move to GREEN is named there (the harness reads them on each rotation).
- Deletes: `ModelsConfig`, `BudgetsConfig`, `TurnsConfig`, `EffortConfig`, `TimeoutsConfig`, `BackendsConfig` (66 role fields); `UnblockAutoConfig`'s six role knobs; `watcher_model`, `watcher_rotation_budget_usd`, `watcher_max_turns`, `watcher_effort`, `watcher_backend`; `simple_task_budget_usd`/`simple_task_max_turns` and their migrator `_migrate_simple_task_scalars`; `routing.py::_config_key`; the inline `startswith('reviewer')` in `TaskWorkflow._invoke`; the 9 `hasattr`/`getattr` lookups in `resolve_route` and the two defaulting `getattr`s in `_invoke`.
- Observable signal: boundary rows 9, 9b and 11 (row 11 from γ2's direct read; δ moves it onto the decision). A live dispatch's `routing_decision` event is unchanged for every role. `check-config` on reify's yaml (knob-major spelling) reports its `budgets.*`/`max_turns.*`/`effort.*`/`timeouts.*` entries as deprecations, not unknown keys, and resolves the same values.
- G7: INV-5/INV-9 (one orientation); INV-11 (the watcher's model is now allowlist-checked); heuristic 3.

**δ — The route resolver and the allowlist move to shared; orchestrator.routing keeps the availability probe** (`task_kind=normal`, priority medium, est. ~1,200 LOC)
- Files (31): `shared/src/shared/route_policy.py` (new), `shared/src/shared/route_resolver.py` (new), `shared/tests/test_route_resolver.py` (new), `orchestrator/src/orchestrator/routing.py`, `orchestrator/src/orchestrator/config.py`, `orchestrator/src/orchestrator/routing_dispatch.py`, `orchestrator/src/orchestrator/workflow.py`, `orchestrator/src/orchestrator/harness.py`, `orchestrator/src/orchestrator/steward.py`, `orchestrator/src/orchestrator/review_checkpoint.py`, `orchestrator/src/orchestrator/dry_run_unblock.py`, `orchestrator/src/orchestrator/evals/runner.py`, `orchestrator/tests/_orch_helpers.py`, `orchestrator/tests/test_dry_run_unblock.py`, `orchestrator/tests/test_eval_driver.py`, `orchestrator/tests/test_model_overrides_end_to_end.py`, `orchestrator/tests/test_out_of_band_routing.py`, `orchestrator/tests/test_routing.py`, `orchestrator/tests/test_routing_byte_equivalence.py`, `orchestrator/tests/test_routing_config.py`, `orchestrator/tests/test_routing_dispatch.py`, `orchestrator/tests/test_routing_integration_gate.py`, `orchestrator/tests/test_routing_one_step_bounds.py`, `orchestrator/tests/test_routing_reload.py`, `orchestrator/tests/test_routing_resolver.py`, `orchestrator/tests/test_routing_retry_tier_up.py`, `orchestrator/tests/test_routing_simple_saturated.py`, `orchestrator/tests/test_routing_theta_plan_shape.py`, `orchestrator/tests/test_scope_capacity_gate_integration.py`, `orchestrator/tests/test_workflow_invocation_end_truthful.py`, `orchestrator/tests/test_workflow_routing_decision.py`.
- Keys: `fk-dac0a2a67201` (the orchestrator half; the run's `routing.py` cycle).
- Deps: γ2; 6766 (S25 λ, C4); 8373 (routing split condition; also through α); 5366 (fail-closed ceiling; also through β1); S20 δ (S20 C6, the invocation cost floor).
- Work: D9, C5, C6. Every `resolve_route` caller passes `config.routing` and its role row. `RoleDefaults` and `PlanShape` imports move to shared. `invocation_bounds` is applied in `routing_dispatch` after `resolve_route`. `OrchestratorConfig.routing` is typed `OrchestratorRoutingPolicy`, whose `rules` default is the stock table, so a project yaml with a partial `routing:` block keeps the stock rules as it does today through the yaml deep-merge. `validate_role_models` also refuses a `Backend.CLAUDE` row whose `budget_usd` is below `shared.claude_argv.INVOCATION_COST_FLOOR_USD` (S20 C6), so a budget that every invocation would refuse fails at load instead.
- Deletes: `routing.py::resolve_route`, `_rule_matches`, `_model_rejection_reason`, `_resolve_ladder_relative`, `_task_simple_saturated`, `RouteInputs`, `RoutingDecision`, `RoleDefaults`, `PlanShape`, `DEFAULT_ALLOWED_MODELS`, `DEFAULT_LADDER`, and its `TYPE_CHECKING` import of `OrchestratorConfig`; `config.py`'s `RuleMatch`/`RuleSet`/`RoutingRule`/`RoutingConfig` definitions (now imports) and its import from `orchestrator.routing`.
- Observable signal: boundary rows 10 and 11. Live `routing_decision` events on dark-factory after deploy carry the same fields plus `backend`/`timeout_secs`, and `scripts/audit_model_admission.py` still attributes rejections (6766's signal holds).
- G7: INV-5 (one resolver); heuristic 13 (the cycle goes: `routing.py` imports `config.py` at runtime).

**ε1 — Recon agent and judge providers are Backend; the dead SDK arms go** (`task_kind=normal`, priority medium, est. ~950 LOC, mostly deletion)
- Files (11): `fused-memory/src/fused_memory/config/schema.py`, `fused-memory/src/fused_memory/reconciliation/agent_loop.py`, `fused-memory/src/fused_memory/reconciliation/judge.py`, `fused-memory/src/fused_memory/reconciliation/prompts/judge.py`, `fused-memory/config/config.yaml`, `fused-memory/tests/test_agent_loop.py`, `fused-memory/tests/test_judge.py`, `fused-memory/tests/reconciliation/harness_fixtures.py`, `fused-memory/tests/reconciliation/test_verify_agent_failure.py`, `fused-memory/tests/test_targeted.py`, `fused-memory/tests/test_stages.py`.
- Keys: `fk-34d8da6424f4`, `fk-709d4d84ee37`.
- Deps: β2 (same file `config/schema.py`), γ0; S12 α (`reconciliation/judge.py`, `tests/test_judge.py`). `config.yaml`'s tail 9194 comes through β2.
- Work: D11. The two provider fields are typed `Backend` through `parse_backend` and pivot to `roles.recon_agent.backend`/`roles.recon_judge.backend` once ε2a lands the table; until then they keep their paths. AgentLoop accepts only `CLAUDE`, and the judge accepts only `CLAUDE` (OQ-3, ruled by the lead). Both are validated at load. The shared recon test fixture `tests/reconciliation/harness_fixtures.py` (imported by 17 test files) sets `agent_llm_provider: 'anthropic'`, which this leaf's `{CLAUDE}` restriction rejects; it moves to `claude_cli`, and the three other tests that configure the SDK providers move with it.
- Deletes: AgentLoop's `_call_llm` SDK arms, `_call_openai`, `_OpenAIResponseAdapter`, `_TextBlock`, `_ToolUseBlock`; `Judge._call_llm`'s OpenAI `else`; its anthropic arm, `_parse_verdict`'s text arm and the prompt's prose output block; the tests that pin those paths. (The mutation-journaling path with `CircuitBreakerError` is already gone from main.)
- Observable signal: boundary row 12. The running fused-memory reconciles a cycle whose judge verdict lands in the recon report as before (live `claude_cli` path unchanged).
- G7: INV-11 (an unknown provider is a load error, never an OpenAI call); INV-9 (one verdict contract home, OQ-3).

**ε2a — Recon LLM calls declare a role and receive a validated route** (`task_kind=normal`, priority medium, est. ~1,100 LOC)
- Files (11): `fused-memory/src/fused_memory/config/schema.py`, `fused-memory/config/config.yaml`, `fused-memory/src/fused_memory/reconciliation/harness.py`, `fused-memory/src/fused_memory/reconciliation/cli_stage_runner.py`, `fused-memory/src/fused_memory/reconciliation/agent_loop.py`, `fused-memory/src/fused_memory/reconciliation/judge.py`, `fused-memory/src/fused_memory/reconciliation/verify.py`, `fused-memory/tests/test_config_schema.py`, `fused-memory/tests/test_harness.py`, `fused-memory/tests/reconciliation/test_cli_stage_runner_cutover.py`, `fused-memory/tests/reconciliation/harness_fixtures.py`.
- Keys: `fk-dac0a2a67201`.
- Deps: δ, ε1; 9188 (S3 ε1, the tail of `cli_stage_runner.py`); S12 β1 (the run driver `drive` that the `_select_tier` rewrite rebases onto). `reconciliation/harness.py`'s tail 9254 comes through β2.
- Work: D10 for `RECON_AGENT`, `RECON_JUDGE`, `RECON_STAGE_STANDARD` and `RECON_STAGE_BACKLOG`. The `model_routing` field and the complete `roles` table land here, with one `TableLegacy` row map on `roles` for every fused-memory old spelling (`reconciliation.sonnet_model` → `roles.recon_stage_standard.model`, `reconciliation.agent_llm_provider` → `roles.recon_agent.backend`, and so on). The four recon rows move their home into the table. Until ε2b, the five non-recon rows are computed by a model validator from the old fields, which stay the only home of those values; each row is derived, never stored twice, so `validate_role_models` covers all nine from this leaf on. `_select_tier` builds `RouteInputs.for_role(...)` and returns the stage role's route, and `cli_stage_runner` takes `budget_usd` from it. The shared recon fixture names model `sonnet`. The `roles` field is `Reload.GREEN` with a `RestartOnlyRows` marker naming `memory_llm`, the four recon rows, `curator` and `path_scope_adjudicator`, today's tiers (D10); ε2a re-runs boundary row 6 against its base list and names each tier change in its commit. The description tells operators: no fused-memory `reload_config` between this merge and fused-memory's next restart (D6's fused-memory residual).
- Deletes: `agent_llm_model`, `judge_llm_model`, `sonnet_model`, `opus_model` as fields (legacy spellings remain); `max_budget_usd=5.0`.
- Observable signal: boundary rows 13 and 13b against the live `fused-memory/config/config.yaml`. After deploy, a recon cycle's stage invocation logs the role's model and budget.
- G7: INV-11; INV-2 (the refusal names role, model and allowlist).

**ε2b — Curator, adjudicators, write-triage judge and the memory LLM take their route from the role table** (`task_kind=normal`, priority medium, est. ~1,100 LOC)
- Files (21): `fused-memory/src/fused_memory/config/schema.py`, `fused-memory/src/fused_memory/middleware/task_curator.py`, `fused-memory/src/fused_memory/middleware/path_scope_adjudicator.py`, `fused-memory/src/fused_memory/server/write_triage_judge.py`, `fused-memory/src/fused_memory/maintenance/link_adjudicator.py`, `fused-memory/src/fused_memory/backends/graphiti_client.py`, `fused-memory/src/fused_memory/backends/mem0_client.py`, `fused-memory/src/fused_memory/routing/classifier.py`, `fused-memory/src/fused_memory/routing/router.py`, `fused-memory/src/fused_memory/server/main.py`, `fused-memory/src/fused_memory/services/overreach_shadow.py`, `fused-memory/src/fused_memory/arm_harness/arm_config.py`, `fused-memory/config/config.yaml`, `fused-memory/config/config-docker.yaml`, `fused-memory/scripts/replay_overreach_extraction.py`, `fused-memory/tests/server/test_write_triage_judge.py`, `fused-memory/tests/test_task_curator.py`, `fused-memory/tests/test_path_scope_adjudicator.py`, `fused-memory/tests/test_link_adjudicator.py`, `fused-memory/tests/arm_harness/test_arm_config.py`, `fused-memory/tests/test_replay_overreach_extraction.py`.
- Keys: `fk-dac0a2a67201` (closing).
- Deps: ε2a; 9204 (S10 δ, `task_curator.py`); 9203 (S26 μ, `graphiti_client.py` and `mem0_client.py`); S19 δ (also through β2).
- Work: D10 for the remaining five roles. `fused-memory/scripts/replay_overreach_extraction.py` and its test, new since the draft, read `config.llm.model`/`.provider` and move with the other readers. `fused-memory/config/config-docker.yaml` sets `llm.provider` and `llm.model` and is rewritten here with `config.yaml`, so ξ's refusal finds nothing in this repository. Their homes move into the table and the deriving validator goes. Each consumer resolves its route per call; a row leaves `RestartOnlyRows` only when ε2b shows its consumer resolves per call (row 3), and `memory_llm` stays restart-only because Graphiti's and Mem0's clients fix the model at construction. `LLMConfig`'s cross-field validation of `client_class` and `structured_output_mode` against the provider moves to a `FusedMemoryConfig` validator over the `memory_llm` row. ε2b re-runs boundary row 6 against its base list and names each tier change in its commit. The `llm` block's model and provider leave `LLMConfig` for `roles.memory_llm` (OQ-7, lead), so every reader of `config.llm.model`/`.provider` moves: `graphiti_client.py`, `mem0_client.py`, `routing/classifier.py`, `routing/router.py`, `server/main.py`, `services/overreach_shadow.py`, `arm_harness/arm_config.py` and `write_triage_judge.py`. The write-triage judge's inheritance from the memory LLM is retired (OQ-8, ruled by the lead): its row has its own default. The description tells operators: no fused-memory `reload_config` between this merge and fused-memory's next restart (D6's fused-memory residual).
- Deletes: `CuratorConfig.model`/`max_budget_usd`, `PathScopeAdjudicatorConfig.model`/`max_budget_usd`, `WriteTriageConfig.judge_provider`/`judge_model`, `LinkHealConfig.adjudicator_model` and `LLMConfig.model`/`provider` as fields (legacy spellings remain); `write_triage_judge.py::resolve_judge_provider`/`resolve_judge_model` (OQ-8).
- Observable signal: boundary row 14. The running write-triage judge's next call logs provider `openai` and model `gpt-6.1-sol` (unchanged), now from `roles.write_triage_judge`.
- G7: INV-9; heuristic 3 (one routing mechanism for the factory's LLM traffic).

**η — Service-path knobs leave ReconciliationConfig; fused-memory consumers trust the validated schema** (`task_kind=normal`, priority low, est. ~1,000 LOC, ~570 of it a moved seed table)
- Files (15): `fused-memory/src/fused_memory/config/schema.py`, `fused-memory/src/fused_memory/config/topic_guard_seed.py` (new), `fused-memory/src/fused_memory/server/near_duplicate_guard.py`, `fused-memory/src/fused_memory/services/topic_anchor.py`, `fused-memory/src/fused_memory/middleware/done_provenance_authz.py`, `fused-memory/src/fused_memory/middleware/task_interceptor.py`, `fused-memory/src/fused_memory/services/memory_mutations.py`, `fused-memory/src/fused_memory/services/graph_mutations.py`, `fused-memory/src/fused_memory/services/memory_service.py`, `fused-memory/config/config.yaml`, `docs/task-authoring.md`, `fused-memory/tests/config_factory.py` (new), `fused-memory/tests/test_topic_anchor.py`, `fused-memory/tests/server/test_near_duplicate_guard.py`, `fused-memory/tests/test_entity_mint_service.py`.
- Keys: `fk-c520d5374ba5`, `fk-a3dfde6f0664`.
- Deps: ε2b (same file `schema.py`); 9174 (S26 η, `topic_anchor.py`); 9245 (S26 τ, `graph_mutations.py`); 9247 (S26 υ, `memory_mutations.py`); 9282 (S10 ν, the tail of `middleware/task_interceptor.py`). `memory_service.py`'s tail 9252 and S26 λ 9194 (the mint lock timeout fold) come through β2.
- Work: D12. The one operator-prose mention of a moved knob (`docs/task-authoring.md`, `reconciliation.deterministic_provenance_allowed_agent_prefixes`) names its new path. The description tells operators: no fused-memory `reload_config` between this merge and fused-memory's next restart (D6's fused-memory residual).
- Deletes: the seven service-path fields from `ReconciliationConfig`; the three `_reconciliation_attr` copies; `_DEFAULT_TOPIC_ANCHOR_ENABLED`; `near_duplicate_guard`'s restated default; the `isinstance`/`getattr` fallbacks in the two storm detectors; `_ENTITY_MINT_DEFAULT_LOCK_TIMEOUT_SECONDS` if λ left it; `_default_topic_guard_clusters` from `schema.py`.
- Observable signal: boundary row 15. On the running server, a `reload_config` that toggles `search_anchor.enabled` changes the next `search`'s pinning, and `add_memory` near-dup refusals continue with the live threshold.
- G7: INV-10 (tests build real configs); INV-11 (an invalid knob is a load error, never a silently disabled alarm); heuristic 6.

**ζ1 — Harness supervision knobs become submodels: service restart keyed by service, fleet redeploy, watcher, digest, escalation revalidation, info-L0 router** (`task_kind=normal`, priority medium, est. ~1,400 LOC)
- Files (38): `orchestrator/src/orchestrator/config.py`, `orchestrator/src/orchestrator/config_sections/supervision.py` (new), `escalation/src/escalation/authority.py`, `fused-memory/src/fused_memory/reconciliation/orphaned_recon_escalation_sweep.py`, `orchestrator/src/orchestrator/agents/briefing.py`, `orchestrator/src/orchestrator/harness.py`, `orchestrator/src/orchestrator/service_restart.py`, `orchestrator/src/orchestrator/workflow.py`, `scripts/orchestrator-watchdog.py`, `orchestrator/src/orchestrator/config_sections/__init__.py` (new), `orchestrator/tests/test_config_shape_golden.py` (new), `orchestrator/tests/fixtures/config_shape/` (new: dark-factory and reify yaml copies), `orchestrator/tests/test_briefing_reviewer_info_notes.py`, `orchestrator/tests/test_code_default_hatch_guard.py`, `orchestrator/tests/test_config.py`, `orchestrator/tests/test_fleet_staleness_composition.py`, `orchestrator/tests/test_harness_digest_chain_telemetry.py`, `orchestrator/tests/test_harness_digest_rollup.py`, `orchestrator/tests/test_harness_escalation_revalidation_sweep.py`, `orchestrator/tests/test_harness_fleet_drain.py`, `orchestrator/tests/test_harness_lane_record_divergence.py`, `orchestrator/tests/test_harness_park_stop.py`, `orchestrator/tests/test_harness_parked_loop.py`, `orchestrator/tests/test_harness_pause_visibility.py`, `orchestrator/tests/test_harness_service_restart.py`, `orchestrator/tests/test_harness_starvation_watchdog.py`, `orchestrator/tests/test_harness_terminal_lane_record_reclaim.py`, `orchestrator/tests/test_harness_watcher_supervisor.py`, `orchestrator/tests/test_info_l0_router_harness.py`, `orchestrator/tests/test_info_l0_router_workflow.py`, `orchestrator/tests/test_merge_queue_warm_cold_shadow.py`, `orchestrator/tests/test_orphan_l0_reaper.py`, `orchestrator/tests/test_service_restart.py`, `orchestrator/tests/test_stale_escalation_cleanup.py`, `tests/scripts/test_orchestrator_restart_config_drift.py`, `tests/scripts/test_orchestrator_watchdog.py`, `orchestrator/tests/_info_l0_router_fixtures.py`, `orchestrator/tests/conftest.py`.
- Keys: `fk-dae2a050b5d3`.
- Deps: δ; 9173 (S3 κ, `escalation/authority.py`); 9251 (S10 κ2, `orphaned_recon_escalation_sweep.py`); 9272 (S5 ρ0, the tail of `service_restart.py`; 9263 is its ancestor); S24 γ and S24 ζ (S24 γ deletes the lease field; S24 ζ re-defaults `watch_prefixes`). `harness.py`'s tail 9236 comes through β1 and `agents/briefing.py`'s 9235 through γ2.
- Work: D5 (ζ1 row), D6. S24 ζ's restart-only `fused_memory_restart_min_interval_secs` moves into the fused-memory row as `min_interval_secs`, unmarked (restart-only), and its field default keeps naming S24's `FM_ON_MERGE_MIN_INTERVAL_SECS` in `shared/fleet_coordination.py`. Each moved field keeps its reload marker and declares its old spelling (D6). Every reader in the listed files moves, the `getattr(config, '<old>', …)` string reads and the `model_copy(update={'<old>': …})` keys included (β1 pivots an old key, so a missed `model_copy` stays correct, but the leaf rewrites them anyway). Boundary row 18's golden map gains the leaf's fields. The listed files are every file that names a moved field as an attribute, a string key, a keyword argument or a yaml key (reader scan at `ba17934861`, re-run at `ce6673c7e1`); `dark-factory-orchestrator.yaml` is κ's, and the per-module `orchestrator.yaml` files keep their flat override names (D5). The one restart default S24 homes in `shared/fleet_coordination.py` (`watch_prefixes`) stays a field default naming that constant.
- Deletes: 19 restart flat fields (19 at `ce6673c7e1`, less the lease field S24 γ deletes, plus the interval S24 ζ adds), and the 10 watcher, 5 digest, 2 escalation-revalidation and 4 info-L0 flat fields.
- Observable signal: rows 3 and 18. After deploy, the two watchdog drift tests (`tests/scripts/test_orchestrator_watchdog.py`, `tests/scripts/test_orchestrator_restart_config_drift.py`) are re-anchored to the new paths and green; the watchdog itself reads no config path (it keeps stdlib mirrors plus env overrides).
- G7: INV-6 and INV-7 are untouched (no behaviour change); INV-11 via D6.

**ζ2 — Recovery sweeps become one submodel per sweep** (`task_kind=normal`, priority medium, est. ~1,100 LOC)
- Files (38): `orchestrator/src/orchestrator/config.py`, `orchestrator/src/orchestrator/config_sections/recovery_sweeps.py` (new), `orchestrator/src/orchestrator/harness.py`, `orchestrator/src/orchestrator/verify.py`, `orchestrator/src/orchestrator/workflow.py` (prose), `orchestrator/tests/test_config_shape_golden.py`, `orchestrator/tests/conftest.py`, `orchestrator/tests/test_atomic_train_merge.py`, `orchestrator/tests/test_code_default_hatch_guard.py`, `orchestrator/tests/test_config.py`, `orchestrator/tests/test_convert_to_blocked.py`, `orchestrator/tests/test_cross_task_dedupe_reengagement.py`, `orchestrator/tests/test_deterministic_task.py`, `orchestrator/tests/test_halt_owner.py`, `orchestrator/tests/test_harness_action_dispatch.py`, `orchestrator/tests/test_harness_cancel_workflow.py`, `orchestrator/tests/test_harness_deterministic_recon_sweep.py`, `orchestrator/tests/test_harness_force_release_wedged_slot.py`, `orchestrator/tests/test_harness_ground_truth_wiring.py`, `orchestrator/tests/test_harness_interactive_reaper.py`, `orchestrator/tests/test_harness_lifecycle_registry.py`, `orchestrator/tests/test_harness_main_tip_sweep.py`, `orchestrator/tests/test_harness_no_landings_breaker.py`, `orchestrator/tests/test_harness_terminal_lane_record_reclaim.py`, `orchestrator/tests/test_harness_terminal_status_watcher.py`, `orchestrator/tests/test_harness_warm_lane_gc.py`, `orchestrator/tests/test_merge_queue_warm_cold_shadow.py`, `orchestrator/tests/test_orphan_l0_reaper.py`, `orchestrator/tests/test_orphan_reaper.py`, `orchestrator/tests/test_reconcile_stranded.py`, `orchestrator/tests/test_recovery_emission_wiring.py`, `orchestrator/tests/test_repend_state_machine.py`, `orchestrator/tests/test_stale_escalation_cleanup.py`, `orchestrator/tests/test_stranded_blocked_sweep.py`, `orchestrator/tests/test_stranded_verified_green.py`, `orchestrator/tests/test_verify_main_tip_sweep.py`, `orchestrator/tests/_info_l0_router_fixtures.py`, `orchestrator/tests/test_info_l0_router_boundary.py`.
- Keys: `fk-dae2a050b5d3`.
- Deps: ζ1; 9160 (S4 ε, `verify.py`); S17 ζ (it deletes `stranded_blocked_redispatch_enabled`, so ζ2 moves 28 fields, not 29, and does not edit `scheduler.py`). Task 4651's landed-reconciler knobs, if 4651 has landed (it precedes 9188), join `recovery_sweeps` here.
- Work: D5 (ζ2 row). The re-pend loop cap that task 4082 adds beside `stranded_reconcile_interval_secs` (S17's ruling) joins `recovery_sweeps.stranded`; the reader scan re-run at filing finds it. Each moved field keeps its reload marker and declares its old spelling (D6). Every reader in the listed files moves, the `getattr(config, '<old>', …)` string reads and the `model_copy(update={'<old>': …})` keys included (β1 pivots an old key, so a missed `model_copy` stays correct, but the leaf rewrites them anyway). Boundary row 18's golden map gains the leaf's fields. The listed files are every file that names a moved field as an attribute, a string key, a keyword argument or a yaml key (reader scan at `ba17934861`, re-run at `ce6673c7e1`); `dark-factory-orchestrator.yaml` is κ's, and the per-module `orchestrator.yaml` files keep their flat override names (D5). `workflow.py` carries one docstring mention (`orphan_l0_merge_phase_freshness_secs`) that is rewritten.
- Deletes: the 28 orphan-L0/stranded/main-tip-sweep/terminal-status/no-landings/warm-lane-GC/worktree-reaper/deterministic-recon-sweep flat fields and `lane_stale_report_days`'s root spelling.
- Observable signal: rows 3 and 18; after deploy, each sweep's first log line on dark-factory reports its interval from the new path.
- G7: INV-11 via D6.

**ζ3 — The scheduler submodel, and the two double-duty knobs get their own fields** (`task_kind=normal`, priority medium, est. ~1,000 LOC)
- Files (35): `orchestrator/src/orchestrator/config.py`, `orchestrator/src/orchestrator/config_sections/scheduler.py` (new), `orchestrator/src/orchestrator/harness.py`, `orchestrator/src/orchestrator/scheduler.py`, `orchestrator/src/orchestrator/backfill_admission.py` (created by S17; reader scan re-run at filing), `orchestrator/src/orchestrator/warm_base_watchdog.py` (created by S17; reader scan re-run at filing), `orchestrator/src/orchestrator/dispatch_ledger.py` (created by S17; reader scan re-run at filing), `orchestrator/src/orchestrator/tick_phases.py` (created by S17; reader scan re-run at filing), `orchestrator/src/orchestrator/delivered_check_gate.py` (created by S17; reader scan re-run at filing), `orchestrator/tests/test_config_shape_golden.py`, `orchestrator/tests/_orch_helpers.py`, `orchestrator/tests/conftest.py`, `orchestrator/tests/test_cascade_unblock.py`, `orchestrator/tests/test_config.py`, `orchestrator/tests/test_config_scoring_reload.py`, `orchestrator/tests/test_cross_project_dispatch_integration.py`, `orchestrator/tests/test_harness_abandoned_merge_deferred_wiring.py`, `orchestrator/tests/test_harness_park_stop.py`, `orchestrator/tests/test_harness_reblock_guard.py`, `orchestrator/tests/test_harness_scheduler_seam.py`, `orchestrator/tests/test_resume_claimant_liveness.py`, `orchestrator/tests/test_retry_cap.py`, `orchestrator/tests/test_scheduler.py`, `orchestrator/tests/test_scheduler_callbacks.py`, `orchestrator/tests/test_scheduler_park_backfill.py`, `orchestrator/tests/test_scheduler_pin_reservations.py`, `orchestrator/tests/test_scheduler_scoring.py`, `orchestrator/tests/test_scheduler_starvation.py`, `orchestrator/tests/test_scheduler_state.py`, `orchestrator/tests/test_scheduler_tool_error_reply.py`, `orchestrator/tests/test_transient_requeue_backoff.py`, `orchestrator/tests/test_warm_lane_structural_exhaustion.py`, `orchestrator/tests/test_workflow_state_machine_boundary.py`, `orchestrator/tests/test_workflow_terminal_report.py`, `orchestrator/tests/test_zero_progress_requeue.py`.
- Keys: `fk-dae2a050b5d3`, `fk-da71e5bb1679`.
- Deps: ζ2; 9234 (S8 μ1, the tail of `scheduler.py`); S17 κ1 (S17's last leaf on `scheduler.py`).
- Work: D5 (ζ3 row), D13. Each moved field keeps its reload marker and declares its old spelling (D6). Every reader in the listed files moves, the `getattr(config, '<old>', …)` string reads and the `model_copy(update={'<old>': …})` keys included (β1 pivots an old key, so a missed `model_copy` stays correct, but the leaf rewrites them anyway). Boundary row 18's golden map gains the leaf's fields. The listed files are every file that names a moved field as an attribute, a string key, a keyword argument or a yaml key (reader scan at `ba17934861`, re-run at `ce6673c7e1`); `dark-factory-orchestrator.yaml` is κ's, and the per-module `orchestrator.yaml` files keep their flat override names (D5). Each moved field's tier is decided and declared, including the four requeue/dispatch scalars task 4602 asks about. The readers S17 moved out of `scheduler.py` move too: `backfill_admission.py` and `warm_base_watchdog.py` (S17 θ), `dispatch_ledger.py` (S17 η), `tick_phases.py` (S17 ε) and `delivered_check_gate.py` (S17 δ2), which reads `max_checks_per_tick`. The two double-duty readers of D13 move wherever S17 left them. Re-run the reader scan at ζ3's filing over the tree S17 κ1 left, and declare what it finds (OQ-11, ruled by the lead). A local, parameter or attribute that holds a submodel never has a name ending in `config` or `cfg` (write `sched`, `self._sched_knobs`).
- Deletes: 22 flat scheduler fields; the "dedicated cap would need a new config.py field" and "deferred (config.py is out of scope)" comments.
- Observable signal: row 17; after deploy, `check-config` lists `scheduler.max_local_backfill_unresolved_cycles` and `delivered_checks.max_record_fetches_per_tick` with today's values.
- G7: INV-11 via D6; heuristic 6 (one knob, one behaviour).

**ζ4a — Verify admission, clock-stop and cgroup knobs become submodels** (`task_kind=normal`, priority medium, est. ~800 LOC)
- Files (47): `orchestrator/src/orchestrator/config.py`, `orchestrator/src/orchestrator/config_sections/verify_admission.py` (new), `orchestrator/src/orchestrator/harness.py`, `orchestrator/src/orchestrator/verify.py`, `scripts/merge-pytest-n-ab-analysis.py`, `orchestrator/tests/test_config_shape_golden.py`, `orchestrator/tests/_task_verify_integration_rig.py`, `orchestrator/tests/conftest.py`, `orchestrator/tests/test_config.py`, `orchestrator/tests/test_config_verify_admission_reload.py`, `orchestrator/tests/test_config_verify_cgroup_cpu_weight.py`, `orchestrator/tests/test_harness_host_pool.py`, `orchestrator/tests/test_harness_lifecycle_registry.py`, `orchestrator/tests/test_merge_gate_rerun_end_to_end.py`, `orchestrator/tests/test_merge_verify_config_fidelity.py`, `orchestrator/tests/test_operational_config_binding.py`, `orchestrator/tests/test_review_checkpoint_phase1_spawn.py`, `orchestrator/tests/test_run_node_ids.py`, `orchestrator/tests/test_task_verify_host_race.py`, `orchestrator/tests/test_task_verify_integration.py`, `orchestrator/tests/test_verify.py`, `orchestrator/tests/test_verify_admission_call_sites.py`, `orchestrator/tests/test_verify_admission_cancel_release.py`, `orchestrator/tests/test_verify_admission_executor_bypass.py`, `orchestrator/tests/test_verify_admission_integration_gate.py`, `orchestrator/tests/test_verify_admission_order_integration.py`, `orchestrator/tests/test_verify_admission_pytest_n.py`, `orchestrator/tests/test_verify_admission_wiring.py`, `orchestrator/tests/test_verify_cancel_unwind.py`, `orchestrator/tests/test_verify_clock_stop.py`, `orchestrator/tests/test_verify_env_recovery_attempt.py`, `orchestrator/tests/test_verify_env_transient.py`, `orchestrator/tests/test_verify_event_channel_wiring.py`, `orchestrator/tests/test_verify_hosts.py`, `orchestrator/tests/test_verify_load_stamp.py`, `orchestrator/tests/test_verify_plan_latch.py`, `orchestrator/tests/test_verify_pytest_n_operator_cap.py`, `orchestrator/tests/test_verify_ruff_config_boundary.py`, `orchestrator/tests/test_verify_scope_cpu_weight.py`, `orchestrator/tests/test_verify_scope_memory.py`, `orchestrator/tests/test_verify_scope_spawn_failure.py`, `orchestrator/tests/test_verify_segmented_fallback.py`, `orchestrator/tests/test_verify_summary_facts.py`, `tests/scripts/test_fallback_verify_config.py`, `tests/scripts/test_merge_gate_parallelism_config.py`, `orchestrator/tests/test_verify_relocated_scope_reap.py`, `tests/scripts/test_verify_budget_census_regimes.py`.
- Keys: `fk-dae2a050b5d3`.
- Deps: ζ3. `verify.py`'s tail 9160 comes through ζ2 and `harness.py`'s through β1.
- Work: D5 (ζ4a row). Each moved field keeps its reload marker and declares its old spelling (D6). Every reader in the listed files moves, the `getattr(config, '<old>', …)` string reads and the `model_copy(update={'<old>': …})` keys included (β1 pivots an old key, so a missed `model_copy` stays correct, but the leaf rewrites them anyway). Boundary row 18's golden map gains the leaf's fields. The listed files are every file that names a moved field as an attribute, a string key, a keyword argument or a yaml key (reader scan at `ba17934861`, re-run at `ce6673c7e1`); `dark-factory-orchestrator.yaml` is κ's, and the per-module `orchestrator.yaml` files keep their flat override names (D5).
- Deletes: the 10 verify_admission_*, 6 verify_clock_stop_*, 3 verify_cgroup_cpu_weight_* flat fields and `verify_use_cgroup_scope`.
- Observable signal: rows 3 and 18; after deploy, the admission controller's startup line on dark-factory reports the same slots and order from `verify_admission.*`.
- G7: INV-11 via D6.

**ζ6a — Workflow iteration caps and review cadence become the workflow submodel** (`task_kind=normal`, priority medium, est. ~900 LOC)
- Files (96): `orchestrator/src/orchestrator/config.py`, `orchestrator/src/orchestrator/config_sections/workflow.py` (new), `orchestrator/src/orchestrator/evals/runner.py`, `orchestrator/src/orchestrator/implementer_brief_variant.py`, `orchestrator/src/orchestrator/merge_lane/gates.py`, `orchestrator/src/orchestrator/workflow.py`, `shared/src/shared/cli_result.py` (prose; S20 β creates it and moves is_zero_output_timeout there), `orchestrator/tests/test_config_shape_golden.py`, `orchestrator/tests/_review_loop_harness.py`, `orchestrator/tests/_task_verify_integration_rig.py`, `orchestrator/tests/_workflow_helpers.py`, `orchestrator/tests/test_architect_all_committed_plan.py`, `orchestrator/tests/test_atomic_train_merge.py`, `orchestrator/tests/test_block_disposition.py`, `orchestrator/tests/test_config.py`, `orchestrator/tests/test_config_pre_merge_reverify_reload.py`, `orchestrator/tests/test_eval_boundary_suite.py`, `orchestrator/tests/test_eval_collect_metrics.py`, `orchestrator/tests/test_eval_live_fixture.py`, `orchestrator/tests/test_eval_profile.py`, `orchestrator/tests/test_eval_recovery.py`, `orchestrator/tests/test_execute_server_error_requeue.py`, `orchestrator/tests/test_liveness_boundary_gate.py`, `orchestrator/tests/test_merge_queue.py`, `orchestrator/tests/test_merge_queue_bounce.py`, `orchestrator/tests/test_merge_queue_deep_integration_gate.py`, `orchestrator/tests/test_one_step_variant_boundary.py`, `orchestrator/tests/test_plan_revalidation_skip.py`, `orchestrator/tests/test_plan_text_fp_preserving_stamps.py`, `orchestrator/tests/test_rebase_branch_reset_guard.py`, `orchestrator/tests/test_rebase_verify_cost.py`, `orchestrator/tests/test_repend_state_machine.py`, `orchestrator/tests/test_restart_recut_e2e.py`, `orchestrator/tests/test_review_cap_exit.py`, `orchestrator/tests/test_review_completed_event.py`, `orchestrator/tests/test_review_context_complete_wiring.py`, `orchestrator/tests/test_reviewer_verdict_routing.py`, `orchestrator/tests/test_routing_integration_gate.py`, `orchestrator/tests/test_steward_outcome.py`, `orchestrator/tests/test_suggestion_triage.py`, `orchestrator/tests/test_verdict_servers_integration_gate.py`, `orchestrator/tests/test_verify_events_units.py`, `orchestrator/tests/test_verify_green_checkpoint.py`, `orchestrator/tests/test_verify_phase_rebase.py`, `orchestrator/tests/test_workflow.py`, `orchestrator/tests/test_workflow_already_done.py`, `orchestrator/tests/test_workflow_amendment.py`, `orchestrator/tests/test_workflow_amendment_wip_guard.py`, `orchestrator/tests/test_workflow_architect_l0_promotion.py`, `orchestrator/tests/test_workflow_architect_transient_retry.py`, `orchestrator/tests/test_workflow_blocking_dep.py`, `orchestrator/tests/test_workflow_cancellation.py`, `orchestrator/tests/test_workflow_dry_run_hook.py`, `orchestrator/tests/test_workflow_e2e.py`, `orchestrator/tests/test_workflow_escalation_warning.py`, `orchestrator/tests/test_workflow_factory.py`, `orchestrator/tests/test_workflow_false_premise.py`, `orchestrator/tests/test_workflow_finalise_without_merge_sha.py`, `orchestrator/tests/test_workflow_first_enqueue_stamp.py`, `orchestrator/tests/test_workflow_infra_thrash.py`, `orchestrator/tests/test_workflow_invocation_end_truthful.py`, `orchestrator/tests/test_workflow_merge_gating_strand.py`, `orchestrator/tests/test_workflow_merge_phase_liveness.py`, `orchestrator/tests/test_workflow_merge_refuses_unmergeable_branch.py`, `orchestrator/tests/test_workflow_merge_retry_resume_merge_entry.py`, `orchestrator/tests/test_workflow_merge_thrash.py`, `orchestrator/tests/test_workflow_metadata_files_reconcile.py`, `orchestrator/tests/test_workflow_narrowing_exit_reports.py`, `orchestrator/tests/test_workflow_plan_repair_replan.py`, `orchestrator/tests/test_workflow_preexisting_main_break.py`, `orchestrator/tests/test_workflow_prompt_builder.py`, `orchestrator/tests/test_workflow_ready_to_merge.py`, `orchestrator/tests/test_workflow_resettled_suppression.py`, `orchestrator/tests/test_workflow_resume_on_progress.py`, `orchestrator/tests/test_workflow_review_persistence.py`, `orchestrator/tests/test_workflow_routing_decision.py`, `orchestrator/tests/test_workflow_signature_loop_guard.py`, `orchestrator/tests/test_workflow_state_machine_boundary.py`, `orchestrator/tests/test_workflow_task_root.py`, `orchestrator/tests/test_workflow_task_verify_port.py`, `orchestrator/tests/test_workflow_terminal_report.py`, `orchestrator/tests/test_workflow_train_attribution.py`, `orchestrator/tests/test_workflow_train_completion.py`, `orchestrator/tests/test_workflow_train_former.py`, `orchestrator/tests/test_workflow_train_halt_owner.py`, `orchestrator/tests/test_workflow_train_state_escalation.py`, `orchestrator/tests/test_workflow_train_thrash_guard.py`, `orchestrator/tests/test_workflow_truthful_commit_stamp.py`, `orchestrator/tests/test_workflow_truthful_exits.py`, `orchestrator/tests/test_workflow_unactionable.py`, `orchestrator/tests/test_workflow_verify_infra_resume.py`, `orchestrator/tests/test_workflow_verify_retry.py`, `orchestrator/tests/test_workflow_worktree_missing.py`, `orchestrator/tests/test_workflow_zero_output_hang.py`, `orchestrator/tests/_info_l0_router_fixtures.py`, `orchestrator/tests/test_workflow_scope_requests.py`.
- Keys: `fk-dae2a050b5d3`.
- Deps: ζ4a; 9153 (S4 ν, `merge_lane/gates.py`); S20 β (it moves `is_zero_output_timeout`, whose docstring names `max_consecutive_zero_output_timeouts`, to `shared/cli_result.py`). `evals/runner.py`'s tail 9235 comes through γ2.
- Work: D5 (ζ6a row). Each moved field keeps its reload marker and declares its old spelling (D6). Every reader in the listed files moves, the `getattr(config, '<old>', …)` string reads and the `model_copy(update={'<old>': …})` keys included (β1 pivots an old key, so a missed `model_copy` stays correct, but the leaf rewrites them anyway). Boundary row 18's golden map gains the leaf's fields. The listed files are every file that names a moved field as an attribute, a string key, a keyword argument or a yaml key (reader scan at `ba17934861`, re-run at `ce6673c7e1`); `dark-factory-orchestrator.yaml` is κ's, and the per-module `orchestrator.yaml` files keep their flat override names (D5). `shared/cli_result.py`, where S20 β moves `is_zero_output_timeout`, names `max_consecutive_zero_output_timeouts` only in that function's docstring (in `shared/cli_invoke.py` at `ce6673c7e1`); ζ6a rewrites the docstring there and no config import enters shared. ζ6a no longer declares `shared/cli_invoke.py`: its only reference moves out in S20 β, which ζ6a depends on (S20's ask). A local, parameter or attribute that holds a submodel never has a name ending in `config` or `cfg` (write `sched`, `self._sched_knobs`).
- Deletes: the 16 iteration-cap and review-cadence flat fields.
- Observable signal: rows 3 and 18; a live task's workflow on dark-factory after deploy runs with identical caps (its `max_review_cycles` read is logged at plan time as today).
- G7: INV-11 via D6.

**ζ6b — Steward knobs become the steward submodel; rebase, revalidation and cleanup knobs join the workflow submodel** (`task_kind=normal`, priority medium, est. ~900 LOC)
- Files (93): `orchestrator/src/orchestrator/config.py`, `orchestrator/src/orchestrator/config_sections/steward.py` (new), `orchestrator/src/orchestrator/evals/profile.py`, `orchestrator/src/orchestrator/exit_contract.py`, `orchestrator/src/orchestrator/harness.py`, `orchestrator/src/orchestrator/steward.py`, `orchestrator/src/orchestrator/workflow.py`, `orchestrator/tests/test_config_shape_golden.py`, `orchestrator/tests/_lane_test_helpers.py`, `orchestrator/tests/_workflow_helpers.py`, `orchestrator/tests/conftest.py`, `orchestrator/tests/test_architect_all_committed_plan.py`, `orchestrator/tests/test_config.py`, `orchestrator/tests/test_config_plan_text_fp_reload.py`, `orchestrator/tests/test_config_pre_merge_reverify_reload.py`, `orchestrator/tests/test_config_reload_integration_gate.py`, `orchestrator/tests/test_conftest_helpers.py`, `orchestrator/tests/test_crash_recovery.py`, `orchestrator/tests/test_done_phase_wedge.py`, `orchestrator/tests/test_eval_boundary_suite.py`, `orchestrator/tests/test_eval_profile.py`, `orchestrator/tests/test_exit_contract.py`, `orchestrator/tests/test_harness_force_release_wedged_slot.py`, `orchestrator/tests/test_harness_warm_lane_wiring.py`, `orchestrator/tests/test_info_l0_router_workflow.py`, `orchestrator/tests/test_lane_checkout_reconcile_conflict.py`, `orchestrator/tests/test_lane_in_use.py`, `orchestrator/tests/test_liveness_boundary_gate.py`, `orchestrator/tests/test_mark_blocked_outcome_scoping.py`, `orchestrator/tests/test_merge_queue.py`, `orchestrator/tests/test_plan_revalidation_skip.py`, `orchestrator/tests/test_plan_text_fp_preserving_stamps.py`, `orchestrator/tests/test_rebase_branch_reset_guard.py`, `orchestrator/tests/test_rebase_verify_cost.py`, `orchestrator/tests/test_repend_state_machine.py`, `orchestrator/tests/test_restart_durable_guards.py`, `orchestrator/tests/test_simple_task_workflow.py`, `orchestrator/tests/test_steward.py`, `orchestrator/tests/test_steward_dismiss_race.py`, `orchestrator/tests/test_steward_outcome.py`, `orchestrator/tests/test_steward_session_cost.py`, `orchestrator/tests/test_suggestion_triage.py`, `orchestrator/tests/test_task_verify_lease.py`, `orchestrator/tests/test_verify_admission_call_sites.py`, `orchestrator/tests/test_verify_events_units.py`, `orchestrator/tests/test_verify_green_checkpoint.py`, `orchestrator/tests/test_verify_phase_rebase.py`, `orchestrator/tests/test_workflow.py`, `orchestrator/tests/test_workflow_already_done.py`, `orchestrator/tests/test_workflow_amendment.py`, `orchestrator/tests/test_workflow_amendment_wip_guard.py`, `orchestrator/tests/test_workflow_architect_l0_promotion.py`, `orchestrator/tests/test_workflow_architect_transient_retry.py`, `orchestrator/tests/test_workflow_blocking_dep.py`, `orchestrator/tests/test_workflow_chronic_flake.py`, `orchestrator/tests/test_workflow_claimant.py`, `orchestrator/tests/test_workflow_dry_run_hook.py`, `orchestrator/tests/test_workflow_e2e.py`, `orchestrator/tests/test_workflow_escalated_steward_stall.py`, `orchestrator/tests/test_workflow_escalation_warning.py`, `orchestrator/tests/test_workflow_factory.py`, `orchestrator/tests/test_workflow_false_premise.py`, `orchestrator/tests/test_workflow_first_enqueue_stamp.py`, `orchestrator/tests/test_workflow_infra_thrash.py`, `orchestrator/tests/test_workflow_invocation_end_truthful.py`, `orchestrator/tests/test_workflow_merge_gating_strand.py`, `orchestrator/tests/test_workflow_merge_phase_liveness.py`, `orchestrator/tests/test_workflow_merge_retry_pending.py`, `orchestrator/tests/test_workflow_merge_retry_resume_merge_entry.py`, `orchestrator/tests/test_workflow_merge_superseded.py`, `orchestrator/tests/test_workflow_merge_thrash.py`, `orchestrator/tests/test_workflow_no_plan_cycle.py`, `orchestrator/tests/test_workflow_prompt_builder.py`, `orchestrator/tests/test_workflow_ready_to_merge.py`, `orchestrator/tests/test_workflow_resettled_suppression.py`, `orchestrator/tests/test_workflow_resume_on_progress.py`, `orchestrator/tests/test_workflow_routing_decision.py`, `orchestrator/tests/test_workflow_state_machine_boundary.py`, `orchestrator/tests/test_workflow_task_verify_port.py`, `orchestrator/tests/test_workflow_train_attribution.py`, `orchestrator/tests/test_workflow_train_completion.py`, `orchestrator/tests/test_workflow_train_former.py`, `orchestrator/tests/test_workflow_train_halt_owner.py`, `orchestrator/tests/test_workflow_train_state_escalation.py`, `orchestrator/tests/test_workflow_train_thrash_guard.py`, `orchestrator/tests/test_workflow_truthful_commit_stamp.py`, `orchestrator/tests/test_workflow_truthful_exits.py`, `orchestrator/tests/test_workflow_unactionable.py`, `orchestrator/tests/test_workflow_verify_infra_resume.py`, `orchestrator/tests/test_workflow_worktree_missing.py`, `orchestrator/tests/test_workflow_zero_output_hang.py`, `orchestrator/tests/_info_l0_router_fixtures.py`, `orchestrator/tests/test_workflow_scope_requests.py`.
- Keys: `fk-dae2a050b5d3`.
- Deps: ζ6a; 9284 (S9 ν, the tail of `exit_contract.py`). `steward.py`'s tail 9229 comes through α.
- Work: D5 (ζ6b row). Each moved field keeps its reload marker and declares its old spelling (D6). Every reader in the listed files moves, the `getattr(config, '<old>', …)` string reads and the `model_copy(update={'<old>': …})` keys included (β1 pivots an old key, so a missed `model_copy` stays correct, but the leaf rewrites them anyway). Boundary row 18's golden map gains the leaf's fields. The listed files are every file that names a moved field as an attribute, a string key, a keyword argument or a yaml key (reader scan at `ba17934861`, re-run at `ce6673c7e1`); `dark-factory-orchestrator.yaml` is κ's, and the per-module `orchestrator.yaml` files keep their flat override names (D5). A local, parameter or attribute that holds a submodel never has a name ending in `config` or `cfg` (write `sched`, `self._sched_knobs`).
- Deletes: the 16 steward_*, suggestion-triage, rebase_*, revalidation_*, terminal-cleanup, claimant-heartbeat, exit-contract and worktree-identity-guard flat fields.
- Observable signal: rows 3 and 18; after deploy, a steward invocation's budget line reports `steward.lifetime_budget` with today's value.
- G7: INV-11 via D6.

**ζ4b — Verify timeouts, retries, cold commands and concurrency become the verify submodel; per-module overrides map onto it** (`task_kind=normal`, priority medium, est. ~1,100 LOC)
- Files (65): `orchestrator/src/orchestrator/config.py`, `orchestrator/src/orchestrator/config_sections/verify.py` (new), `orchestrator/src/orchestrator/merge_lane/drift.py`, `orchestrator/src/orchestrator/merge_lane/worker.py`, `orchestrator/src/orchestrator/task_verify_run.py`, `orchestrator/src/orchestrator/task_verify_spec.py`, `orchestrator/src/orchestrator/verify.py`, `orchestrator/src/orchestrator/verify_runner.py`, `orchestrator/src/orchestrator/verify_spec_shared.py`, `orchestrator/src/orchestrator/workflow.py`, `orchestrator/tests/test_config_shape_golden.py`, `orchestrator/tests/_orch_helpers.py`, `orchestrator/tests/_task_verify_fixtures.py`, `orchestrator/tests/_task_verify_integration_rig.py`, `orchestrator/tests/_verify_host_drive.py`, `orchestrator/tests/test_atomic_train_merge.py`, `orchestrator/tests/test_cli_verify_task.py`, `orchestrator/tests/test_config.py`, `orchestrator/tests/test_conftest_helpers.py`, `orchestrator/tests/test_main_tip_sweep_provisioned_env.py`, `orchestrator/tests/test_merge_drift.py`, `orchestrator/tests/test_merge_gate_rerun_end_to_end.py`, `orchestrator/tests/test_merge_gates_typecheck.py`, `orchestrator/tests/test_merge_lane_restart_drain.py`, `orchestrator/tests/test_merge_queue.py`, `orchestrator/tests/test_merge_queue_coalesce.py`, `orchestrator/tests/test_merge_queue_deep_integration_gate.py`, `orchestrator/tests/test_merge_queue_equivalence.py`, `orchestrator/tests/test_merge_queue_verify_retry.py`, `orchestrator/tests/test_merge_verify_config_fidelity.py`, `orchestrator/tests/test_multihost_verify_integration.py`, `orchestrator/tests/test_remote_verify_dispatch.py`, `orchestrator/tests/test_review_checkpoint_phase1_spawn.py`, `orchestrator/tests/test_run_node_ids.py`, `orchestrator/tests/test_task_verify_host_race.py`, `orchestrator/tests/test_task_verify_integration.py`, `orchestrator/tests/test_task_verify_run.py`, `orchestrator/tests/test_task_verify_spec.py`, `orchestrator/tests/test_timeout_marker_inversion_guard.py`, `orchestrator/tests/test_train_integration.py`, `orchestrator/tests/test_verify.py`, `orchestrator/tests/test_verify_admission_integration_gate.py`, `orchestrator/tests/test_verify_admission_order_integration.py`, `orchestrator/tests/test_verify_cancel_unwind.py`, `orchestrator/tests/test_verify_categories.py`, `orchestrator/tests/test_verify_cold_preprovision.py`, `orchestrator/tests/test_verify_env_recovery_attempt.py`, `orchestrator/tests/test_verify_env_transient.py`, `orchestrator/tests/test_verify_event_channel_wiring.py`, `orchestrator/tests/test_verify_events_units.py`, `orchestrator/tests/test_verify_pyright_interpreter_misresolution.py`, `orchestrator/tests/test_verify_ruff_config_boundary.py`, `orchestrator/tests/test_verify_runner.py`, `orchestrator/tests/test_verify_scope_inversion_boundary.py`, `orchestrator/tests/test_verify_scope_kappa.py`, `orchestrator/tests/test_verify_sequential_lint_first.py`, `orchestrator/tests/test_verify_spec_shared.py`, `orchestrator/tests/test_workflow_task_verify_port.py`, `orchestrator/tests/test_workflow_verify_infra_resume.py`, `tests/scripts/test_fallback_verify_config.py`, `tests/scripts/test_module_verify_budgets.py`, `tests/scripts/test_pyright_version_pin.py`, `tests/scripts/test_scripts_module_config.py`, `tests/scripts/test_tests_scripts_module_config.py`, `orchestrator/tests/test_verify_relocated_scope_reap.py`.
- Keys: `fk-dae2a050b5d3`.
- Deps: ζ6b; 8528 (R55 parking: declares `merge_lane/worker.py`); 9188 (S3 ε1, the tail of `worker.py`); 9166 (S4 ρ, `drift.py` and `verify_runner.py`); 9135 (S4 η, `task_verify_spec.py` and `verify_spec_shared.py`). `task_verify_run.py`'s tail 9150 comes through γ2.
- Work: D5 (ζ4b row and the module layer). Each moved field keeps its reload marker and declares its old spelling (D6). Every reader in the listed files moves, the `getattr(config, '<old>', …)` string reads and the `model_copy(update={'<old>': …})` keys included (β1 pivots an old key, so a missed `model_copy` stays correct, but the leaf rewrites them anyway). Boundary row 18's golden map gains the leaf's fields. The listed files are every file that names a moved field as an attribute, a string key, a keyword argument or a yaml key (reader scan at `ba17934861`, re-run at `ce6673c7e1`); `dark-factory-orchestrator.yaml` is κ's, and the per-module `orchestrator.yaml` files keep their flat override names (D5). `_OVERRIDABLE_FIELDS` becomes `MODULE_OVERRIDE_TARGETS: Mapping[str, str]` beside `ModuleConfig`, mapping each flat module key to its root path, with an import-time check that every target resolves in `OrchestratorConfig`. Task 9135 (S4 η) deletes `apply_spec_to_host_config`, folding it into `verify_spec_shared.py::apply_merge_verify_spec`, which returns an `AppliedSpec`. ζ4b adds `verify_spec_shared.py::apply_config_overrides`, which applies the spec's overrides by dotted path onto submodel copies, and `apply_merge_verify_spec` calls it, so the `AppliedSpec`'s config carries them at their nested paths; the remote-verify spec's own dict keys stay unchanged, because they are a wire format with back-compat `d.get(...)` defaults. `merge_lane/worker.py` reads `verify_command_timeout_secs` through a string `getattr`, which moves.
- Deletes: the 11 verify timeout, retry, cold-command, drift-cadence and concurrency flat fields; `_OVERRIDABLE_FIELDS`.
- Observable signal: rows 3, 18, 19 and 20; after deploy, a dark-factory task verify reports the same command timeout and a remote verify on the laptop applies the same cold pre-provision command.
- G7: INV-11 via D6 (an unresolvable module target fails at import).

**ζ5a — Merge-lane knobs become the merge_lane submodel** (`task_kind=normal`, priority medium, est. ~1,200 LOC)
- Files (97): `orchestrator/src/orchestrator/config.py`, `orchestrator/src/orchestrator/config_sections/merge_lane.py` (new), `orchestrator/src/orchestrator/harness.py`, `orchestrator/src/orchestrator/merge_lane/gates.py`, `orchestrator/src/orchestrator/merge_lane/history_leg.py`, `orchestrator/src/orchestrator/merge_lane/worker.py`, `orchestrator/src/orchestrator/verify.py`, `orchestrator/src/orchestrator/verify_cmd.py`, `orchestrator/src/orchestrator/verify_plan.py`, `orchestrator/src/orchestrator/verify_runner.py`, `orchestrator/src/orchestrator/workflow.py`, `scripts/verify_budget_census.py`, `orchestrator/src/orchestrator/verify_spec_shared.py`, `orchestrator/tests/test_config_shape_golden.py`, `fused-memory/tests/fixtures/pack_pool_2026-11-04.json`, `orchestrator/tests/_merge_lane_fakes.py`, `orchestrator/tests/_orch_helpers.py`, `orchestrator/tests/_task_verify_integration_rig.py`, `orchestrator/tests/conftest.py`, `orchestrator/tests/test_atomic_train_merge.py`, `orchestrator/tests/test_config.py`, `orchestrator/tests/test_config_disjoint_skip_reload.py`, `orchestrator/tests/test_config_merge_gate_history_pure.py`, `orchestrator/tests/test_harness.py`, `orchestrator/tests/test_harness_k_from_config.py`, `orchestrator/tests/test_harness_parked_loop.py`, `orchestrator/tests/test_hermetic_disk_usage.py`, `orchestrator/tests/test_lane_scene_config_guard.py`, `orchestrator/tests/test_liveness_boundary_gate.py`, `orchestrator/tests/test_merge_boundary_effective_module_configs.py`, `orchestrator/tests/test_merge_gate_already_on_main.py`, `orchestrator/tests/test_merge_gate_rerun_end_to_end.py`, `orchestrator/tests/test_merge_gate_reverify_host.py`, `orchestrator/tests/test_merge_gates_typecheck.py`, `orchestrator/tests/test_merge_history_leg_runner.py`, `orchestrator/tests/test_merge_lane_cancel_request.py`, `orchestrator/tests/test_merge_lane_coalesce_admission.py`, `orchestrator/tests/test_merge_lane_coalesce_unwind.py`, `orchestrator/tests/test_merge_lane_instrumentation.py`, `orchestrator/tests/test_merge_lane_intake_buffers.py`, `orchestrator/tests/test_merge_lane_relane.py`, `orchestrator/tests/test_merge_lane_restart_drain.py`, `orchestrator/tests/test_merge_queue.py`, `orchestrator/tests/test_merge_queue_coalesce.py`, `orchestrator/tests/test_merge_queue_deep_integration_gate.py`, `orchestrator/tests/test_merge_queue_disposition_wiring.py`, `orchestrator/tests/test_merge_queue_equivalence.py`, `orchestrator/tests/test_merge_queue_flake_recorder.py`, `orchestrator/tests/test_merge_queue_lifecycle_registry.py`, `orchestrator/tests/test_merge_queue_main_health.py`, `orchestrator/tests/test_merge_queue_orphan_reaper.py`, `orchestrator/tests/test_merge_queue_store.py`, `orchestrator/tests/test_merge_queue_train_attribution.py`, `orchestrator/tests/test_merge_queue_verify_retry.py`, `orchestrator/tests/test_merge_reverify_gate_record.py`, `orchestrator/tests/test_merge_same_tree_gate.py`, `orchestrator/tests/test_merge_skew_end_to_end.py`, `orchestrator/tests/test_merge_verify_config_fidelity.py`, `orchestrator/tests/test_multihost_verify_integration.py`, `orchestrator/tests/test_rebase_verify_cost.py`, `orchestrator/tests/test_run_node_ids.py`, `orchestrator/tests/test_steward_outcome.py`, `orchestrator/tests/test_train_integration.py`, `orchestrator/tests/test_train_moved_main.py`, `orchestrator/tests/test_verify.py`, `orchestrator/tests/test_verify_cold_preprovision.py`, `orchestrator/tests/test_verify_events_units.py`, `orchestrator/tests/test_verify_plan.py`, `orchestrator/tests/test_verify_preexisting_main_break.py`, `orchestrator/tests/test_verify_runner.py`, `orchestrator/tests/test_verify_scope_inversion_boundary.py`, `orchestrator/tests/test_verify_scope_kappa.py`, `orchestrator/tests/test_verify_scope_lambda.py`, `orchestrator/tests/test_verify_scope_spawn_failure.py`, `orchestrator/tests/test_verify_segmented_fallback.py`, `orchestrator/tests/test_verify_spec_shared.py`, `orchestrator/tests/test_workflow.py`, `orchestrator/tests/test_workflow_finalise_without_merge_sha.py`, `orchestrator/tests/test_workflow_merge_refuses_unmergeable_branch.py`, `orchestrator/tests/test_workflow_merge_retry_resume_merge_entry.py`, `orchestrator/tests/test_workflow_preexisting_main_break.py`, `orchestrator/tests/test_workflow_prompt_builder.py`, `orchestrator/tests/test_workflow_resume_on_progress.py`, `orchestrator/tests/test_workflow_state_machine_boundary.py`, `orchestrator/tests/test_workflow_task_verify_port.py`, `orchestrator/tests/test_workflow_train_former.py`, `orchestrator/tests/test_workflow_truthful_commit_stamp.py`, `orchestrator/tests/test_workflow_verify_infra_resume.py`, `orchestrator/tests/test_workflow_verify_retry.py`, `orchestrator/tests/test_workflow_zero_output_hang.py`, `tests/scripts/test_config_retune_gate_decision.py`, `tests/scripts/test_merge_gate_history_leg.py`, `tests/scripts/test_module_verify_budgets.py`, `tests/scripts/test_scripts_module_config.py`, `orchestrator/tests/test_merge_queue_deep_reservation.py`, `orchestrator/tests/test_workflow_scope_requests.py`, `tests/scripts/test_verify_budget_census_regimes.py`.
- Keys: `fk-dae2a050b5d3`.
- Deps: ζ4b; 8528 (R55 parking: declares `merge_lane/worker.py`); 9188 (`worker.py` tail, through ζ4b). `history_leg.py`'s tail 9153 comes through ζ6a and `verify_cmd.py`/`verify_plan.py`'s 9160 through ζ2.
- Work: D5 (ζ5a row). Each moved field keeps its reload marker and declares its old spelling (D6). Every reader in the listed files moves, the `getattr(config, '<old>', …)` string reads and the `model_copy(update={'<old>': …})` keys included (β1 pivots an old key, so a missed `model_copy` stays correct, but the leaf rewrites them anyway). Boundary row 18's golden map gains the leaf's fields. The listed files are every file that names a moved field as an attribute, a string key, a keyword argument or a yaml key (reader scan at `ba17934861`, re-run at `ce6673c7e1`); `dark-factory-orchestrator.yaml` is κ's, and the per-module `orchestrator.yaml` files keep their flat override names (D5). `merge_lane/gates.py`'s `getattr(config, 'merge_verify_breadth', None) == 'full'` moves to the typed read, so dark-factory's whole-tree merge gate stays whole-tree (boundary row 20). `verify_runner.py` maps the spec's `merge_verify_*` keys onto dotted paths through ζ4b's `apply_config_overrides`, inside S4 η's `apply_merge_verify_spec`. `scripts/verify_budget_census.py` reads the nested path. `max_advance_attempts` is read only by `merge_lane/worker.py`, so it moves here (D5 places "advance attempts" in `merge_lane`). The ratchet runs per the merge-lane baseline's `_README` (`scripts/merge_lane_metrics.py --check`). A local, parameter or attribute that holds a submodel never has a name ending in `config` or `cfg` (write `sched`, `self._sched_knobs`).
- Deletes: the 14 merge_train_*, merge_verify_*, merge_gate_history_*, heartbeat, disjoint-skip, preexisting-break and advance-attempt flat fields.
- Observable signal: rows 3, 18 and 20; after deploy, a dark-factory merge's `merge_attempt` event is unchanged and its gate is still whole-tree.
- G7: INV-11 via D6; INV-12 (the whole-tree gate cannot silently degrade to scoped).

**ζ5b — Verify-host knobs become the verify_hosts submodel** (`task_kind=normal`, priority medium, est. ~600 LOC)
- Files (27): `orchestrator/src/orchestrator/config.py`, `orchestrator/src/orchestrator/config_sections/verify_hosts.py` (new), `orchestrator/src/orchestrator/merge_lane/worker.py`, `orchestrator/src/orchestrator/task_verify_dispatch.py`, `orchestrator/src/orchestrator/verify_runner.py`, `orchestrator/tests/test_config_shape_golden.py`, `orchestrator/tests/_task_verify_integration_rig.py`, `orchestrator/tests/test_concurrent_verify_boundary.py`, `orchestrator/tests/test_config_task_verify_policy.py`, `orchestrator/tests/test_config_verify_host_policy.py`, `orchestrator/tests/test_harness_host_pool.py`, `orchestrator/tests/test_merge_gate_reverify_host.py`, `orchestrator/tests/test_merge_lane_instrumentation.py`, `orchestrator/tests/test_merge_lane_vacated_entry_census.py`, `orchestrator/tests/test_merge_queue_concurrent_verify.py`, `orchestrator/tests/test_merge_queue_deep_integration_gate.py`, `orchestrator/tests/test_merge_queue_dispatch_fill_redispatch.py`, `orchestrator/tests/test_merge_queue_flake_recorder.py`, `orchestrator/tests/test_merge_queue_multihost_wiring.py`, `orchestrator/tests/test_merge_queue_verify_retry.py`, `orchestrator/tests/test_merge_worker_host_health.py`, `orchestrator/tests/test_multihost_verify_integration.py`, `orchestrator/tests/test_task_verify_dispatch.py`, `orchestrator/tests/test_task_verify_integration.py`, `orchestrator/tests/test_train_verify_host.py`, `orchestrator/tests/test_verify_host_skew_bench.py`, `orchestrator/tests/test_merge_queue_deep_reservation.py`.
- Keys: `fk-dae2a050b5d3` (closing).
- Deps: ζ5a; 8528 (R55 parking: declares `merge_lane/worker.py`). `task_verify_dispatch.py`'s tail 9150 comes through γ2.
- Work: D5 (ζ5b row), adopting the host-policy shape S4 (R28) landed. Each moved field keeps its reload marker and declares its old spelling (D6). Every reader in the listed files moves, the `getattr(config, '<old>', …)` string reads and the `model_copy(update={'<old>': …})` keys included (β1 pivots an old key, so a missed `model_copy` stays correct, but the leaf rewrites them anyway). Boundary row 18's golden map gains the leaf's fields. The listed files are every file that names a moved field as an attribute, a string key, a keyword argument or a yaml key (reader scan at `ba17934861`, re-run at `ce6673c7e1`); `dark-factory-orchestrator.yaml` is κ's, and the per-module `orchestrator.yaml` files keep their flat override names (D5). A local, parameter or attribute that holds a submodel never has a name ending in `config` or `cfg` (write `sched`, `self._sched_knobs`).
- Deletes: the 9 verify_host_*, verify_task_* and verify_cross_check_remote_green flat fields.
- Observable signal: rows 3 and 18; after deploy, the remote-verify routing decision for a dark-factory merge is unchanged.
- G7: INV-11 via D6.

**κ — Dark-factory's operator yaml, its dashboard module yaml and its operator docs use the new setting names** (`task_kind=normal`, priority medium, est. ~500 LOC of yaml and prose)
- Files (13): `dark-factory-orchestrator.yaml`, `dashboard/orchestrator.yaml`, `orchestrator/tests/test_config_shape_golden.py`, `OPERATIONS.md`, `ARCHITECTURE.md`, `SETUP.md`, `CLAUDE.md`, `skills/orchestrate/SKILL.md`, `skills/orchestrate/references/project-setup.md`, `skills/escalation-watcher-auto/SKILL.md`, `docs/plan-scoring-and-judge.md`, `docs/task-escalation-state-spec.md`, `docs/config-retune-gate.md`.
- Keys: `fk-dae2a050b5d3`.
- Deps: ζ5b (and through it every ζ leaf and γ2); the clerk's amendment of the yaml-anchored checks in §Delivered checks on moved paths. κ → S12 δ (`CLAUDE.md`, shared with S12 δ's prompt-and-skill sweep; lead ruling, wired at decompose by label). The other prose files are shared with other wave-3 plans; the lead orders them at decompose under the per-file rule.
- Work: (1) rewrite every old name in `dark-factory-orchestrator.yaml` (36 at `ce6673c7e1`) to its new path, reading the map from `shared.config_fields.legacy_spellings(OrchestratorConfig)` rather than a hand list, keeping each comment with its setting, and keeping `lifecycle_refusal_policy` wherever S6 υ is filed to look; (2) move `dashboard/orchestrator.yaml`'s six project-level old names (D5), leaving its module-override keys flat; (3) rewrite the 94 lines of instruction-bearing operator prose in the ten listed files that name a moved orchestrator setting (measured at `ce6673c7e1`; dated records under `docs/ruling-*/`, corpora, PRDs and `plans/` stay as written); (4) the golden test gains the rewritten copy of dark-factory's yaml beside the old-spelled one. The description tells operators the D6 residual: until the dark-factory unit restarts onto ζ5b's code, a `reload_config` on dark-factory reads ζ5b's new spellings as unknown keys, those settings fall back to their defaults, and the census files its level-2 escalation; so avoid that reload until the restart. The lead amends the memories that tell sessions to edit a moved key (lead action).
- Deletes: every old name in this repository's two operator yamls and in its operator prose.
- Observable signal: `orchestrator check-config` on both yamls reports zero deprecated spellings, and the resolved values equal the old-spelled copy's (`model_dump()` diff empty). No declared old name appears in the ten prose files. A `reload_config` on the running orchestrator after its next restart reports empty `applied` and empty `restart_required`.
- G7: INV-11; INV-9 (one spelling per setting in the docs too).

**χ — The escalation watcher spawns the session of a gate Leo pre-authorised, in attended and AFK posture alike** (`task_kind=normal`, `complexity='simple'`, priority medium, est. ~80 LOC of skill text)
- Files (1): `skills/escalation-watcher/SKILL.md`.
- Keys: none (`source: review-all`, `x_finding_run`; it carries Leo's OQ-2 addition).
- Deps: none. The skill file has many pending holders (wave-2 tail 9288, S2 μ, S25 6770): χ edits one
  region (§"Always ask — keyed on record content" and AFK shift 2) and takes S2 μ's stated exception to R54,
  so file locks serialise it and it waits on no holder.
- Why: σ files a `milestone_gate` from the deterministic runner. Today the watcher skill asks Leo about
  every milestone gate (two exceptions only: curator-only gates and `review_due`), and AFK shift 2 forbids
  unattended interactive spawns. A sentence in the gate's description cannot override either: the skill
  keys on record content it names, and a named exception wins over prose. So without χ an AFK watcher
  leaves σ pending, which is the failure Leo's addition exists to avoid.
- Work: add a third exception to the milestone-gate class, keyed on task metadata the watcher reads with
  `get_task`: `metadata.x_preauthorised_spawn = {ruled_by: 'human', ruled_on: '<date>', ruling_ref:
  '<repo path>::<section>', prompt_from: 'description'}`. The watcher checks, as code would, that
  `ruled_by` is `human` and that `ruling_ref` opens to a section quoting the ruling; then it `/spawn`s a
  session whose prompt is the task's description, in attended and AFK posture alike (the one exception to
  AFK shift 2), appends the escalation id to the exclude-file, and never resolves the gate itself: the
  spawned session resumes it on success, or leaves it pending for Leo at its escape row. A failed check
  falls back to "Always ask". S2's identity-header gate (S2 ν) carries the same key and depends on χ
  (lead ruling after the wave-3 fold-verifiers: the mechanism stays in this plan, and S2 ν depends on χ
  because χ has no prerequisite inside this plan).
- Deletes: nothing.
- Observable signal: when σ's escalation fires, the watcher's digest records a spawn for it, whether Leo is
  away or not, and no "ask Leo" line.
- G7: `one-decision-partition-per-area` (INV-23; the check is a predicate over the record; only the spawned session judges);
  INV-11 (a gate without the key is still asked, never silently spawned).

**σ — Operator gate: a pre-authorised /team session rewrites the nine project yamls and the two leo-laptop host yamls to the new setting names** (`task_kind=deterministic`, `always_escalates=true`, no `before_done`, `metadata.milestone = {"mode": "delayed", "after_secs": 259200}`, `metadata.x_preauthorised_spawn` (χ); priority medium)
- Files (0): none in this repository (`metadata.files: []`). The session edits one file in each of nine other repositories and two files on leo-laptop.
- Keys: none (`source: review-all`, `x_finding_run` only).
- Deps: ζ5b (the last orchestrator moving task); κ (the dark-factory yaml rewrite); χ (the watcher rule that spawns it); 6754 (pending: a `GitConfig` rename whose old name reify's yaml sets; it must be declared before the sweep reads the map, D6).
- Shape (the cross-stream operator-sweep shape): 72 hours after ζ5b, κ and χ are done (the delayed milestone, lead ruling: dark-factory's `orchestrator_restart_min_interval_secs` is 259200, so by then the merge-landed restart has usually moved every unit onto ζ5b's code), the gate files a born-at-L2 escalation and goes `blocked`. The escalation watcher spawns the /team session the description carries, under χ's rule, keyed on `metadata.x_preauthorised_spawn = {ruled_by: 'human', ruled_on: '2026-11-12', ruling_ref: 'plans/review-all-s16-orchestrator-config-shape-defaults-and-routing-prd.md::Rulings (sitting of 2026-11-12)', prompt_from: 'description'}`. The session resumes the gate when it finishes, which releases ξ. The description is the brief below, filed verbatim.
- Observable signal: the session's report gives one line per file (commit sha or skip reason, and the `check-config` result). Afterwards `check-config` over every operated yaml lists no deprecated spelling.
- G7: INV-13 (each real file's own `check-config` is the proof); `one-decision-partition-per-area` (INV-23; every step names who decides it).

*σ's brief (the gate's description).* Leo pre-authorised this session on 2026-11-12, in his words: "a L2
task for the watcher to /spawn a pre-authorised /team session to update all the project yamls before the
deploy". **The spawn is pre-authorised and may run while Leo is away**: the task carries
`metadata.x_preauthorised_spawn`, which the escalation-watcher skill's pre-authorised-gate exception (task χ)
keys on, so the watcher spawns this session in AFK posture too. This follows R37's best-effort
cross-repository yaml edit, now as a task. Goal: every operated orchestrator yaml uses only the new setting
names before task ξ makes the old names load errors. The session never edits dark-factory's own files (κ
did) and never restarts the fleet.

| # | Step | Who decides |
|---|---|---|
| 1 | Deployment check. `scripts/orchestrator-watchdog.py --report` lists each running unit's code sha; for each unit, `git merge-base --is-ancestor <ζ5b's merge commit> <unit sha>`. Old code ignores a new name, so no file is edited while any unit lags. | Code. If a unit lags: stop before editing, report the lagging units, and leave the escalation pending for Leo (the delayed milestone makes this rare; a unit whose restart was deferred by a busy merge is the usual cause). |
| 2 | The old-to-new map: `shared.config_fields.legacy_spellings(OrchestratorConfig)` at main. | Code. |
| 3 | Rewrite each file in the table below: each old name to its new path, each comment kept with its setting. | The session (a mechanical edit). |
| 4 | Verify each file: `orchestrator check-config <file>` reports no deprecated spelling, no unknown key and no conflict, and the resolved config before and after the rewrite is equal (`model_dump()` diff empty). | Code. |
| 5 | Commit: in each repository, one yaml-only `git commit --only dark-factory-orchestrator.yaml` on its main; record the sha. | The session. A repository whose main cannot take a yaml-only commit (dirty file, feature branch, hook refusal) is skipped with its reason. |
| 6 | leo-laptop: edit `~/.config/orchestrator/dark-factory-laptop.yaml` and `reify-laptop.yaml` in place (they are in no repository), keep a dated backup beside each, and run step 4 there. | The session; step 4 decides. |
| 7 | Report one line per file on the escalation, then resume the gate (`resolve_issue(action='resume')`) when every file is committed or recorded as having no old name. The watcher never resolves it (χ). | Code (the report and the all-files predicate). |
| — | Escape: a file that fails step 4, an unreachable host, or a skipped file that still carries old names. | Leo: that file is reported, and the gate stays pending. |

Files, with their old names as measured on 2026-11-12 at `07aaa5be80` (no project yaml was edited after
2026-11-05; the session recounts from step 2's map): reify 26 (`budgets`, `max_turns`, `effort`, `timeouts.*`,
`unblock_auto.timeout_seconds`, verify, merge-lane, verify-admission and verify-host keys,
`watcher_daily_cost_ceiling_usd`, `digest_ewa_threshold`, `workflow_exit_contract_enforce`);
solar-challenge-platform 8; solar-challenge 7 (unit `orchestrator-my-solar-challenge`, yaml at
`/home/leo/src/solar-challenge/dark-factory-orchestrator.yaml`); pump-web-ui 4; stream-touch 4;
scoping-intake 4; know-live 3; autopilot-video 2; market-predictor 2 (no unit; rewritten anyway); autotrade
0 (yaml untracked on a feature branch; recorded, not edited). leo-laptop: `dark-factory-laptop.yaml` 5,
`reify-laptop.yaml` 2. Both laptop files also set `verify_cgroup_memory`; whether that key moved is read
from step 2's map, not assumed.

**ξ — Old setting names are refused at load and at hot reload in both services, with an error naming the new path; the translation path is deleted** (`task_kind=normal`, priority medium, est. ~500 LOC)
- Files (14): `shared/src/shared/config_fields.py`, `shared/src/shared/config_reload.py`, `orchestrator/src/orchestrator/config.py`, `orchestrator/src/orchestrator/cli.py`, `fused-memory/src/fused_memory/config/schema.py`, `fused-memory/src/fused_memory/server/tools.py`, `fused-memory/src/fused_memory/server/tool_families/ops.py`, `shared/tests/test_config_fields.py`, `shared/tests/test_config_reload.py`, `orchestrator/tests/test_config_shape_golden.py`, `orchestrator/tests/test_config_model_copy_pivot.py`, `orchestrator/tests/test_config_unknown_keys.py`, `fused-memory/tests/test_config_schema.py`, `fused-memory/tests/test_config_reload.py`.
- Keys: `fk-dae2a050b5d3`.
- Deps: σ (every operated yaml is rewritten); η (the last fused-memory move, so every fused-memory old name is declared). The rest comes through σ.
- Work: C3's end state (D6 phase 3). Both configs' root validators call `refuse_legacy_spellings`. `pivot_legacy_spellings`, `DeprecatedSpelling`, `ConflictingSpellings` and C4's `deprecated` report key go. `census_config_keys` loses its deprecated bucket, and `check-config` reports a declared old name as refused, with its new path, and exits 1. `OrchestratorConfig.model_copy` raises on a declared old key, naming the new path. Fused-memory's `get_status` loses `config.deprecated_spellings` (wherever S15 placed the tool under `server/`). A reload of a file carrying an old name returns `reloaded: false` with the refusal and applies nothing (C4's rollback). Not `extra='forbid'`. Boundary row 18's golden test flips to refusal; rows 21–23 are added. Before committing, run the refusing `check-config` over the nine project yamls in place (read-only) and the two laptop files (over `ssh leo-laptop`), and record each result in the commit message. A refusal there means σ missed a file: the task stops and escalates, and never edits another repository. A file the task's sandbox cannot read is recorded as unchecked, and σ's report and ω's deployed-fleet signal cover it.
- Deletes: the translation path: `pivot_legacy_spellings`, `DeprecatedSpelling`, `ConflictingSpellings`, the census's deprecated bucket, the reload report's `deprecated` key and fused-memory's `config.deprecated_spellings`. It also replaces the draft's dated pivot-removal follow-up.
- Observable signal: rows 21–23. After deploy, every orchestrator unit and fused-memory start cleanly, because σ rewrote every yaml first. `check-config` on a scratch copy of any project yaml with one old name added exits 1, naming the new path.
- G7: INV-11 (an old name is neither silently dropped nor silently translated); INV-2 (the error names both paths).

**π — The scoped-cap model has one spelling: one shared Fable id, one alias decoder, and scope_for matches through it** (`task_kind=normal`, priority low, est. ~250 LOC)
- Files (10): `shared/src/shared/model_names.py` (new), `shared/tests/test_model_names.py` (new), `shared/src/shared/config_models.py`, `shared/src/shared/account_state.py`, `shared/tests/test_usage_gate_scope.py`, `orchestrator/src/orchestrator/routing.py`, `orchestrator/tests/test_routing.py`, `dark-factory-orchestrator.yaml`, `OPERATIONS.md`, `skills/orchestrate/SKILL.md` (two lines there state the old default literal).
- Keys: none (the lead's decision on the task-8253 investigation's side finding; `source: review-all`, `x_finding_run`).
- Deps: κ (same file `dark-factory-orchestrator.yaml`; through it δ on `routing.py` and β1 on `OPERATIONS.md`); S20 ι (S20 ε creates `shared/account_state.py` with `scope_for`, and S20 ι, which follows ε, rewrites the same file; π edits it after both). S20 ν also edits `account_state.py` and is not chained to π: the lead orders the two at decompose (either direction is cycle-free).
- Work: D15, C9. First, measure which id the CLI reports for a `fable` run (the canonical model of a recent nightly-preparer run, read as `shared/process_cost.py::_entry_model` reads it). `FABLE_MODEL` takes that id, `claude-fable-5-1` unless the measurement disagrees; if it does, the task reports the difference to the lead, because admission names `claude-fable-5-1`. Then: `UsageCapConfig.scoped_cap_models` defaults to `[FABLE_MODEL]`; `scope_for` matches `canonical_model(model)`; `orchestrator/routing.py::FABLE_CANDIDATE_MODEL` is replaced by `FABLE_MODEL` at its readers (`routing.py`, `test_routing.py`); dark-factory's `usage_cap.scoped_cap_models` override and its comment go, since the override would restate the default (the field is restart-tier, so the change takes effect at the next restart); `OPERATIONS.md` §"Model-scoped account caps" names the default and says the alias is decoded. The two frozen eval arms that pass `claude-fable-5` (`evals/configs.py`) are not edited (an eval slate is frozen); the commit message states effect (2) of D15. Task 5407 is cancelled into this task.
- Deletes: the `['claude-fable-5']` default literal; `FABLE_CANDIDATE_MODEL`; dark-factory's `scoped_cap_models` override.
- Observable signal: row 24. After deploy, a Fable cap hit in the nightly preparer records `scope: claude-fable-5-1` on its cost event and leaves the account's general scope open.
- G7: INV-9 (one home for the scoped-model fact); INV-11.

**ω — Integration gate: both services load, reload, refuse old names and route through the shared config contracts** (`task_kind=normal`, priority medium, est. ~450 LOC of tests)
- Files (2): `orchestrator/tests/test_config_shape_integration_gate.py` (new), `fused-memory/tests/test_routing_integration_gate.py` (new).
- Keys: `fk-dae2a050b5d3`, `fk-dac0a2a67201`, `fk-2a67692ff456`.
- Deps: ξ, ι (κ, σ and η come through ξ).
- Work: run boundary rows 1–24 end to end against the real config classes and committed fixture copies of the dark-factory, reify and fused-memory yamls. The translation-era rows (4, 4b, 5, 9b, 13b, 15, 18, 18b) run in the refusal form ξ gave them.
- Deletes: nothing.
- Observable signal: the gate passes in CI. On the deployed fleet, both services' `reload_config` with unchanged yamls returns `reloaded: true` with nothing applied. `check-config` across the twelve known roots and the two remote-verify host yamls on leo-laptop (`~/.config/orchestrator/dark-factory-laptop.yaml`, `reify-laptop.yaml`) lists no old name, no unknown key and no conflict. A scratch yaml with one old name is refused, naming the new path.
- G7: INV-13 (the producer proof is the live fleet's own load and reload).

## Existing tasks (contract §8: steps 1 and 2 run on 2026-11-10 and re-run on 2026-11-13)

- **3038** (pending, low; reverse census check for `defaults.yaml` keys without model fields): C7
  covers it as a merge-gate test (a leaf with no field fails the guard). Lead action: cancel with
  `x_acceptance_reason` naming α.
- **4602** (deferred, low; decide the reload tier of four requeue/dispatch scalars): ζ3 declares every
  moved field's tier, these four included. Lead action: cancel with `x_acceptance_reason` naming ζ3.
- **5407** (pending, low; `usage_cap.scoped_cap_models` names the stale `claude-fable-5` literal): it asks
  whether to repoint the default, carry both literals, or leave it. π answers it under the lead's
  one-spelling decision (D15). Lead action: cancel with `x_acceptance_reason` naming π.
- **5109** (pending, medium; a per-module yaml edit is invisible to `reload_config`): fixed against the
  shared engine. Lead action: edge 5109 → β1.
- **6754** (pending, low; rename `git.warm_verify_shadow_compare_nightly_interval_secs`, which reify's yaml
  sets, with a deprecated alias): use C3's `LegacySpelling` instead of a hand alias, and edit the field
  default in `config.py`, never `defaults.yaml`. Lead actions: edge 6754 → β1; amend. σ → 6754 is σ's own
  dependency (decompose), so the rename is declared before σ's sweep reads the map (if 6754 landed after ξ, its old name would be refused on
  deploy and reify would fail to start, D6).
- **8717** (pending, low; partition fused-memory `config/schema.py`): measure after η moves the seed
  table and the service knobs. Lead action: edge 8717 → η.
- **5377** (pending, medium; retire or enable `judge`/`triage`): either order is safe. If it lands
  first, γ2's table has fewer rows. Lead action: amend (edit the field default in `config.py`, never
  `defaults.yaml`).
- **6766**, **8373**, **5366**: δ depends on each (5366 through β1, 8373 through α). **8375**: α depends
  on it and absorbs its rule table. **3792**: β1 depends on it. **8143**, **8452**: β2 depends on each.
  **7434** (`invoke.py` runner seam) and **8376** (curator complexity hint) are rebase-only neighbours of
  γ1 and ε2b.
- **4693** (pending, low; a `digest_ewa_threshold` description fix): rebase-only with ζ1, which moves
  the field.
- **8253** (pending, low; rescoped by Leo on 2026-11-12 to its return-brief half): it no longer edits
  `usage_gate.py`, `config_models.py` or `account_pool.py`, so it shares no file with π. No action.
- **9493** (done; the double-SIGTERM fix) and **9521** (blocked at level 2; the operator reinstall that puts
  9493 in effect): S24's interval flip (S24 θ), which changes the constant ζ1's field default names, waits on
  both (not on S24 ω: the lead exempted θ's one-line `OPERATIONS.md` edit from the same-file chain). Nothing
  in this plan waits on either. No action here.

## Out of scope

- Routing the eval drivers (`orchestrator/evals/*`) and `cli._run_plan_only`'s hard-coded model
  through the resolver (OQ-5, ruled by the lead). Eval arms pin candidate models deliberately, and the CLI
  site is another finding's key.
- A per-backend adapter record factoring the five decisions each `_invoke_<backend>` hand-rolls
  (OQ-6, ruled by the lead). Task 7434 owns the adjacent runner seam.
- Splitting `GitConfig`; fused-memory's own `${VAR}` interpolation copy (tasks 4243, 5853); a
  fused-memory unknown-key census.
- Moving the per-module `orchestrator.yaml` layer onto nested spellings (D5 keeps it flat).
- Refusing every unknown key (`extra='forbid'`): undeclared keys keep today's self-closing census
  escalation (D6).
- `shared/invocation_outcome.py` and `shared/cli_invoke.py`'s backend branches, and which `Backend` members
  are pooled (S20 converts them on C1).
- One allowlist file for both services (OQ-10, ruled by the lead): a follow-up if a rollout shows the two
  drifting.

## Rulings (sitting of 2026-11-12)

Rulings file `digests/wave3-rulings-applied.md` (v2, final; residual answers of 2026-11-13); lead
rulings `digests/wave3-lead-rulings.md`. Every question below is ruled; none is pending.

- **OQ-1** — wait behind the earlier wave file by file (per-file order against filed wave-2 tasks), as
  recommended. Ruled by the lead.
- **OQ-2** (sitting key S16-1) — option B: old setting names are refused at load and at hot reload, with an
  error naming the new path, after one pre-authorised yaml sweep; the refusal starts after the last move
  (residual R-5 = B). Translation stays while the moves land (reported as deprecated, no level-2
  escalation); one human-gate task for the watcher to spawn a pre-authorised /team session that rewrites
  the nine project yamls and the two laptop files (σ); then one small task makes every old name a load
  error (ξ). Not `extra='forbid'`. Ruled by Leo (2026-11-12, R-5 on 2026-11-13). It amends R37: the
  cross-repository yaml edit R37 left as a best-effort lead action is now a task. The recommendation was A
  (translate and report, never refuse).
- **OQ-3** — the judge becomes Claude-CLI-only and `JUDGE_VERDICT_SCHEMA` is the verdict format's one
  home, as recommended; flagged: the SDK provider stops being a valid judge setting, which extends R15's
  deletion. Ruled by the lead.
- **OQ-4** — one dependency-free shared module (C8) holds the few defaults that cross packages, the field
  default names the constant, and the dashboard imports it, as recommended; flagged: it reads R15's
  invariant as allowing a field default that names one shared constant, and it corrects the dashboard's
  lock depth (2 shown, 4 run). S22/OQ-5 is its duplicate, and S24's restart defaults follow the same rule.
  Ruled by the lead.
- **OQ-5** — evaluation drivers and the plan-only CLI path stay outside the router, as recommended. Ruled
  by the lead.
- **OQ-6** — no per-backend adapter record in this plan (task 7434 or a later plan), as recommended. Ruled
  by the lead.
- **OQ-7** — names as drafted, with `memory_llm`. Ruled by the lead.
- **OQ-8** — retire the write-triage judge's inheritance; its row carries today's effective values, as
  recommended; flagged: an operator who changes the main memory model must now set the judge too. Ruled by
  the lead.
- **OQ-9** — one module per subsystem under `orchestrator/config_sections/`, and the clerk widens the
  checks of 3587 and 8528, as recommended. Ruled by the lead.
- **OQ-10** — per-service allowlist overrides on one shared default, as recommended. Ruled by the lead.
- **OQ-11** — this plan waits for the scheduler plan (S17) on the scheduler's settings, as recommended.
  Ruled by the lead.
- **Cross-stream, scoped-cap model names** (from the task-8253 investigation, 2026-11-12) — the three
  spellings of the scoped-cap model become one config point of truth in this plan: C9, D15, task π, which
  absorbs task 5407. Ruled by the lead.
- **Cross-stream, restart settings** (S24-2 = B) — `fused_memory_restart_min_interval_secs` is
  restart-only and ζ1 carries it into the fused-memory row; its default is S24's constant in
  `shared/fleet_coordination.py`, not C8 (the lead's later placement ruling wins over the rulings file's
  "shared defaults module"), and S16's first task takes no edge onto S24. Ruled by Leo (S24-2), placement by
  the lead.
- **Lead rulings after the wave-3 fold-verifiers (S16 block)** — this repository's orchestrator yaml is
  rewritten once, in κ, not inside each move; the prose sweep of the docs and `CLAUDE.md` is in scope (κ;
  κ → S12 δ on `CLAUDE.md`); σ carries a delayed milestone of 72 hours after its prerequisites; no edge
  from α onto S24. Ruled by the lead.
- **Cross-stream, alignment items** (`digests/wave3-alignment-todo.md`) — C1 gains `ZCODE` (task 5399),
  and also `MIMO`, which task 7993 landed the same way; ζ6a drops `shared/cli_invoke.py`. Ruled by the
  lead.
- **Cross-stream, standing rules** (Leo, 2026-11-12) — `one-decision-partition-per-area` (INV-23; σ's step
  table names who decides each row); hermetic evals (this plan adds no store an eval cell can reach).
  Ruled by Leo.

Tactical, decided in the leaf:

1. **Membership edge cases in D5** — a field read by two subsystems goes to the one that executes the
   behaviour it tunes. Decide in each ζ leaf and record the map in its commit.
2. **The census bucket name for deprecated spellings** (`deprecated` vs `legacy`). Decide in β1; ξ removes
   the bucket.
3. **`RoleSettings.effort`: a closed union of the claude, codex and pi effort levels, or `str`.** No
   effort enum exists to reuse. Decide in γ0.
4. **Fixture copies for row 18** — committed snapshots of the dark-factory and reify yamls, refreshed by
   each ζ leaf. Decide in ζ1 whether reify's copy is trimmed to the moved keys.
