# /review-all Remediation Program — dark_factory — 2026-10-10

Program of 26 PRD streams acting on `plans/review-all-dark_factory-20261008.md` (record:
`plans/review-all-dark_factory-20261008.json`, run `review-all-dark_factory-20261008`, pinned at
`6551b75aba6b1885a2aea4bfc46ac82446fba4d4`). This document is the authoritative G4 seam map and
shared convention set for every `/prd` session in the program. Each stream's session MUST read this
file before authoring. Deliberation rulings R1–R24 were made by Leo on 2026-10-10 (R25 onward carry their own date and `ruled_by`) with the run's lead
(`skills/review-all/references/deliberation.md`); they are recorded under Resolved decisions and
are not relitigated unless a finding's evidence changed.

Findings are cited by key (`fk-…`), never by display id or array position. Every key below resolves in
the record's `findings` (this run) or `prior_rows` (the hotspot refresh `hotspot-survey-dark_factory-20261009`
and the review `review-dark_factory-20261009`, both at the same tree).

## Streams

Waves: 1 = agent-mode streams launched now as one `/team`; 2 = high-risk spawn streams (interactive `/prd`
sessions, next sittings); 3 = remaining spawn streams. Upstream deps are seam dependencies from the G4 table:
a stream's tasks that touch an upstream seam declare dependencies on the upstream stream's leaf for it.

| ID | Slug | Scope (one line) | Finding keys | Mode | Wave | Upstream deps |
|----|------|------------------|--------------|------|------|---------------|
| S1 | one-escalation-filing-entry-point | escalation/filing.py: file(record, policy: FilingPolicy{dedup: none|open_at_level(sentinel,level)|fingerprint(DedupeConfig)|root_cause, born_level, sentinel_identity, off | 11 keys: fk-7d1f3a0d12ca, fk-333277fb4bd7, fk-0d31f1114911, fk-32903f2e64e0, fk-2e0a4419290b, fk-cd446ff3da74, fk-9c72e4399b4e, fk-1bf4bfa8be04, fk-b7b0be25c6b4, fk-0ee68c0948ab, fk-1b573c2428c2 | spawn | 3 | — |
| S2 | typed-escalation-record-facts | escalation/models.py as the typed vocabulary: Status/Severity/Category/ResolveAction StrEnums with OPEN/TERMINAL subsets; Escalation gains cascade_parent_id, filing_site, | 14 keys: fk-8c48269c7363, fk-5332d25b1702, fk-e3b7459c681d, fk-d0e1094c99cb, fk-bbb0a7b3c8a8, fk-60b7c33ea0ad, fk-638cca2f65bb, fk-479375564c2f, fk-755f786c32be, fk-48a76ee9f746, fk-ece88b7661b7, fk-9c7ebc2b50b2, fk-a65731424439, fk-89388166d3ca | spawn | 3 | S1 |
| S3 | escalation-server-and-queue-boundary | Split create_server along the seat's partition: ladder tools registered from an escalation module taking (mcp, queue, dedupe_config, lookups); merge/ops tools registered  | 15 keys: fk-a8d3003e9d83, fk-e903cc072b15, fk-43f431f697a3, fk-609a1832903b, fk-4a0626fc033d, fk-5a9e713f4055, fk-e5d81cc1f1d4, fk-07be43f313bc, fk-f8e9e0da3786, fk-66746efddd4d, fk-bc64099a1cd0, fk-ce2dc99e0305, fk-43bad4e724da, fk-c8c3406fd7f4, fk-0de72a91f89a | spawn | 3 | S6, S2 |
| S4 | verify-run-context-and-runner-contract | Frozen VerifyContext(resource_class, worktree_provenance, verdict_authority, retry_owner, module_registry) with constructors merge_gate()/main_probe()/task()/background() | 24 keys: fk-0bb03714372a, fk-5037fafc755c, fk-0f9cace0cd00, fk-fac2cb6af858, fk-d99f146ff901, fk-0c5e73f13485, fk-8dbae7d39adc, fk-e906c4e914c2, fk-5fa21a4b85cb, fk-baa454b81f6f, fk-3756ad2f3357, fk-f994995001e5, fk-caa8f680e1c2, fk-3b67e8761674, fk-09ad1a19f587, fk-a2fb58789c14, fk-c88f7d3ae54a, fk-37d63bf83608, fk-cb29fac47dba, fk-126a75a57ca5, fk-73b03674a8e4, fk-a635d54da4ab, fk-3e806baf4c6e, fk-591206484ea5 | spawn | 2 | — |
| S5 | gitops-ports-query-contract-and-the-git-runner | shared.git_async.run_git is the factory-wide git runner (git_ops._run already adapts it, task 3778): point the 6 src importers of _run (verify, 4 merge_lane modules, esca | 24 keys: fk-b1e3368f99a1, fk-6f7520ff1987, fk-e2ac53b0d72b, fk-d727d82341b3, fk-d176c9ae1cc7, fk-7a3ee979b48a, fk-78b757799cab, fk-a7cefe40ef4c, fk-a7d433cf8308, fk-ad8c0528c5e1, fk-eb7b9f65f412, fk-3ee93d431321, fk-856541951d65, fk-c5d5769d68c6, fk-718c82fcc1a2, fk-739af676767e, fk-c00c5f01be4c, fk-d73aeaebb792, fk-3a4cc6298c02, fk-31e9856ed662, fk-a10d37257808, fk-54eb9f7cc0be, fk-20135ef54f75, fk-c04bf2045ad3 | spawn | 2 | — |
| S6 | merge-lane-lifecycle-outcome-types-and-provenance | ItemLifecycle becomes the operative state (containers derived); one _retire(item, outcome) replaces 21 raw set_result and 31 hand-placed retire/drop/requeue calls | 25 keys: fk-313f404f3cc6, fk-a2984bf98f85, fk-e61eed486dbc, fk-80a790e6f793, fk-c55b12f4b66d, fk-e8a03e4ec067, fk-ba531799522b, fk-1432a52e4753, fk-769c3bb58ce1, fk-153195971a14, fk-e521254a5b71, fk-e7deb8bcfa95, fk-e768d50f8363, fk-83625b943aa5, fk-201d70aaf409, fk-a0f6af37f0d6, fk-59b3df6f52a3, fk-2c9b6486d554, fk-e501862ffe2c, fk-d9e6c03dc259, fk-d41caf6b6331, fk-4b76c9695674, fk-7f2bce2d482c, fk-fd91b14b3ead, fk-75fc1b96098d | spawn | 2 | S5, S7 |
| S7 | merge-lane-gates-placement-mechanisms-and-target-adapters | GateMeasurement (measured(value) | unmeasurable(reason, rc, stderr)) returned by every git-probing gate; the caller decides fail-open/closed in one place and every unmeas | 5 keys: fk-88e3c228b649, fk-689227abfab1, fk-7d1580bb3f03, fk-dd411ada7c29, fk-5d4406afeade | spawn | 3 | — |
| S8 | harness-composition-root-and-slot-registry | Harness as a pure composition root: a SlotRegistry (one frozen per-slot record registered at dispatch, released in one done-callback, deterministic slots included) read b | 19 keys: fk-c1ac3df8e44d, fk-d8f08696dd66, fk-fc3c4cba6a0f, fk-b01dac97e6e9, fk-d0c02dc767dd, fk-1e13b183c961, fk-b1c6965066ed, fk-d470c7828682, fk-17b789fe3ce6, fk-e6a18c90d2fe, fk-bf4197a9a7d9, fk-5cad48d24b6f, fk-5d0a66a2d32e, fk-a1085af3a290, fk-f1a7f57fa58e, fk-a97501641592, fk-538940c8d347, fk-58d1f3489c98, fk-9854bd9a928f | spawn | 2 | S5, S9, S3 |
| S9 | workflow-phase-spine-and-recovery-ledger | Typed artefact models in artifacts.py: PlanState as a derived property, IterationEntry as a discriminated union with is_work decided at the writer, typed architect-exit r | 20 keys: fk-37190b541196, fk-9973521f4f59, fk-108cdd00d122, fk-f37f7e253cd5, fk-2c0594928a18, fk-ee708be4436f, fk-4f7a933c0dd4, fk-a152341e38ed, fk-44bc973b1cc4, fk-d2ba7601c130, fk-bab6058b23c8, fk-e39550272df9, fk-9eb149f0706d, fk-a503ccb37b4b, fk-a71254176a81, fk-b8efaa973162, fk-7d410939a337, fk-67b5971ed72d, fk-e43472e7d5d7, fk-5ea52a0f89ad | spawn | 2 | S10, S4 |
| S10 | task-write-contract-metadata-modes-success-signal-and-registry | One wire contract in shared.task_metadata: metadata_mode Literal['merge','additive','replace'] required with metadata, text_mode Literal['replace','append'] for details/p | 20 keys: fk-ff5234ac1013, fk-116481623cfb, fk-de964b86bdd8, fk-f6224c565a9b, fk-1f8a4c69798f, fk-fa3e3d765f50, fk-f528087f44c7, fk-941b5369a3d7, fk-d80c1c44bca1, fk-d4d61005457b, fk-260352fa900a, fk-cc04e61d23ec, fk-06ab2d3e9148, fk-c41aa02e0779, fk-85e1293e0aa4, fk-1eb04ceac9e8, fk-e3e99447e11e, fk-de3df95984a5, fk-fa355c2c0498, fk-1695d6b02400 | spawn | 2 | — |
| S11 | curator-pipeline-and-candidate-types | middleware/curator_types.py (CandidateTask, RewrittenTask, CuratorDecision, PreparedCandidate, CuratorFailureError, normalize_title, embedding_text) imported downward by  | 7 keys: fk-f61accce2f37, fk-1e0486bf9ab9, fk-4f895cd425c2, fk-caf8bcd5b7ed, fk-756aef0b35b7, fk-85234d9d39c4, fk-b259ba3af605 | agent | 1 | — |
| S12 | reconciliation-run-lifecycle-and-stage-interface | One run driver (start_run → stages → pre-S3 flush → complete_run → verify → judge → except ladder) parameterised by scope/events, used by run_full_cycle, _run_remediation | 21 keys: fk-8621784d79d3, fk-2571f7726184, fk-377a7036ac7f, fk-f23c6e137ce5, fk-0cce86f5dbfe, fk-a3598aec7dca, fk-d2c7f1b902ac, fk-63b65973bd33, fk-37356472614a, fk-90e488a174b7, fk-a88506a1e327, fk-0485729893fe, fk-16baf39b1db6, fk-147a9951edc7, fk-10c8c43d9e89, fk-20cb93d17ff3, fk-9850339b610c, fk-be8ed2cd9ce9, fk-05443fb78da6, fk-2bb6c913d747, fk-875cca636438 | spawn | 3 | S14 |
| S13 | recon-sweeps-flag-identity-and-recon-report-state | A public flag-identity leaf (fingerprint, normalizer, signature, FlagTypeFamily, UUID extraction, composite task-id parsed to a typed subject) stating the persisted-forma | 20 keys: fk-51ec1e4010af, fk-917fdf32a542, fk-5d00812d5256, fk-d9bfbf342b5b, fk-65aeb80b161d, fk-e64ee986eeed, fk-9f66b6f4420c, fk-b1cda781c60e, fk-9165ae3a1e72, fk-90f374fcce21, fk-fa1fd33c7a9e, fk-9b76dd8a1cdb, fk-cdb8c20a3d0f, fk-2868338d3cf3, fk-46514b4ef0cc, fk-fbbf580e4e1d, fk-4c8d19c64d99, fk-e30ecd1f36b8, fk-0791e9817b84, fk-7c13df0b988c | spawn | 3 | S14, S12 |
| S14 | shared-leaves-for-cross-package-facts | Stdlib-only shared leaves imported downward by every package: project_layout.py (runs_db, tasks_db, escalation_queue_dir, scheduler_state, default/known roots; Escalation | 16 keys: fk-06de98fa328b, fk-3644c6841b3e, fk-7a85d17a7c7e, fk-e367d6a83223, fk-1b25c0b5d3a2, fk-43a75be2cd47, fk-f6ab6bfad0be, fk-37f2baab3aec, fk-b84e0aa4e545, fk-f5e16692e170, fk-7aff55dd1465, fk-75fa713ba3d5, fk-f01bf7bc23f9, fk-954f6787b07c, fk-9cdaf62ed9af, fk-ccd26ad88810 | agent | 1 | — |
| S15 | fused-memory-mcp-server-boundary | A ToolContext (interceptor bound once as real or null object, write journal, triage counter, scope parser, read-journal decorator) that per-family registration modules im | 12 keys: fk-a9f21093207e, fk-d085b903f01c, fk-db357b64fcae, fk-22ab8d1f3a42, fk-1cadfcc36cfa, fk-1509bcd45e2a, fk-45cf67bff540, fk-330a60eef271, fk-9214f842ca92, fk-009fbbf855e7, fk-b3d95cdaa0d3, fk-05a87fd1777d | spawn | 3 | — |
| S16 | orchestrator-config-shape-defaults-and-routing | Per-subsystem submodels with reloadability declared on the field (RELOADABLE_FIELDS derived); one role-keyed RoleSettings table with RoleName and Backend as shared StrEnu | 11 keys: fk-dae2a050b5d3, fk-5805ae6f97f1, fk-2a67692ff456, fk-dac0a2a67201, fk-ba569525f6e0, fk-34d8da6424f4, fk-709d4d84ee37, fk-c520d5374ba5, fk-a3dfde6f0664, fk-efae65b1f9af, fk-da71e5bb1679 | spawn | 3 | — |
| S17 | scheduler-seams-and-tick-state | A behaviour-level test builder over acquire_next/release/record_requeue/set_override asserting on get_state_snapshot and events, so the 1,483 private references retire fi | 13 keys: fk-a571802ee4ea, fk-4b35b1adea0d, fk-cc01eb7b143f, fk-3e1e7c5129b5, fk-8af92edd6f38, fk-6c103745c784, fk-b2cd27267851, fk-09bf639213ca, fk-a2bc16e5812c, fk-4b3eb71db017, fk-5a5d08e775de, fk-ebce5e23048a, fk-6e7e9bb942d6 | spawn | 3 | S14, S8 |
| S18 | evals-measurement-integrity | One eval-layer call helper on detect_invocation_error returning Scored|Refused(cause) used by every scoring site, with an explicit unmeasured state reducers exclude and c | 12 keys: fk-0d4c1014e41f, fk-3505aba0481d, fk-28ca5c64de06, fk-43584c88970d, fk-b80c8e52e526, fk-b6d389a4fcf2, fk-2e8daab1913f, fk-294ee8dfa70f, fk-df50bc042300, fk-b2c28b4f8f17, fk-14926fe151d4, fk-4352b1ee49f5 | agent | 1 | — |
| S19 | package-layering-and-private-seams | A declared fused_memory subpackage DAG enforced by a test over the snapshot's typed edges: the 16 write-path recon modules move to fused_memory.recon_state below services | 12 keys: fk-4fb8ffbb0141, fk-420bc8f9a425, fk-64e06774ee0e, fk-d91317f88f21, fk-cfcf00669c46, fk-e8c835c3f412, fk-57599c7fd907, fk-ea4463b19d86, fk-e22732500172, fk-ad5323d27142, fk-99f44f47a6cc, fk-3ddc73250c64 | spawn | 3 | S14 |
| S20 | cli-invocation-and-usage-gate | One classification of an invocation (classify_invocation with strict-confirm as a parameter) consumed by invoke_with_cap_retry, UsageGate.detect_cap_hit and _run_probe; C | 19 keys: fk-1778eb482209, fk-e59c30a25d27, fk-734bc1fc18bf, fk-240df0521f83, fk-dcc5dce60bb6, fk-b8cba955b358, fk-2a5d97aeb64d, fk-8805d20c0c4c, fk-005378a1b462, fk-49b55c1270ae, fk-377f5a42e8be, fk-25c96d016239, fk-f7f1747302dc, fk-db060c262575, fk-ae6cd7600d0e, fk-aacd5e9381f6, fk-3ae32b390a81, fk-fede65204150, fk-2f4b8655fb3c | spawn | 3 | S16 |
| S21 | legibility-pipeline | One detector layer (digest's carrier/provenance/pattern code as its own module) returning per-record class hits that scan_transcript reduces, plus one agent-class classif | 16 keys: fk-3a23d0a51e69, fk-f1314d3b25eb, fk-74142ca0f882, fk-e3cc7c87d73e, fk-15f40965b6ba, fk-da87d818b1b8, fk-988f412eced2, fk-ee9a4bae4e83, fk-ef642b412580, fk-3ea2b2df348a, fk-12f0d08a6ccd, fk-c9ab2773a5d9, fk-0531ecaefba5, fk-0af74b3620ec, fk-234e0a63dfd7, fk-a0af0f46c1dc | agent | 1 | — |
| S22 | dashboard-readers-and-aggregation | One typed label→root resolver (one|unknown|ambiguous) for scheduler, api/tasks, task_prose and config; one FANOUT_FAILURE_EXCEPTIONS classification at every fan-out and o | 22 keys: fk-a9e45ca0eb10, fk-389c36ad02d1, fk-70e1ee9218a7, fk-1c1d247caaa7, fk-b5760d0dfe85, fk-94fc76e7a123, fk-8dfa5e5a75aa, fk-c927adb040e9, fk-1b8c06cd8b2f, fk-ce66d4da9b0f, fk-ad49b80c5554, fk-af4ba730a48a, fk-4afbeb684a22, fk-166cacf84a18, fk-d21f64525e52, fk-9979f1b771de, fk-c51423b7b7e5, fk-d0a7ebc4a4f1, fk-6aee6c49fa2e, fk-f3a5a34831e9, fk-1bdd95e804cb, fk-9cf2a161c9c0 | agent | 1 | S15, S14 |
| S23 | sampler-collection-groups-and-schema | A frozen MetricGroup(name, collect, exact, stems, windowed) table in one module: collectors take names from it, __main__ loops it with one degrade handler and takes it as | 7 keys: fk-e86a10b9e217, fk-37f494b26355, fk-ab2c587ff5a9, fk-d207683eef5a, fk-be1621618dee, fk-fcefc2576467, fk-895f269d7bb5 | agent | 1 | — |
| S24 | fleet-watchdog-and-redeploy-coordination | A frozen per-tier supervision spec (unit source, start property, watched paths, clock path, min interval, redeploy unit/script, lease path, streak/cap, storm escape) cons | 9 keys: fk-906957d2ec84, fk-c9390bf37295, fk-ed19c0c4d4df, fk-3830e2a6de79, fk-68efae8a36fc, fk-1cc71283157c, fk-4c21aaf01b82, fk-47a7eb8c020d, fk-63633270af4e | spawn | 3 | S14 |
| S25 | scripts-instruments-shared-locators-and-parity-checkers | One scripts/_checkout.py bootstrap imported first by every script; locators from shared.project_layout (S14); a run_parity(units, compare, probes) driver in systemd_unit_ | 13 keys: fk-3042baa5983e, fk-638eb09e52b1, fk-b16470a6e0d2, fk-7fee08736560, fk-9a99057011f3, fk-cb9f382ab2bd, fk-679f3dc5ca48, fk-f70004144a3d, fk-30cd73379093, fk-915ca17415b7, fk-9bb2996a620a, fk-195f86e31fea, fk-f4af080fcd25 | agent | 1 | S20, S14 |
| S26 | memory-service-write-path-and-graph-client | Typed queued-write values per operation in a leaf beside durable_queue with to_json/from_json and an operation enum; MemoryService constructed with its always-present col | 18 keys: fk-1215aacfa626, fk-017442f273d2, fk-9430031679c4, fk-eb0d5fea9a52, fk-265caf7191a5, fk-d37d63bf0d11, fk-6e50b22357f2, fk-abc3487a3528, fk-9804ecc4567d, fk-62af616b8470, fk-90f6fbb5daa0, fk-da5b47d0493a, fk-50cc281018fb, fk-343c4a42be6b, fk-bdc0427c8971, fk-41eaf65cc4d0, fk-b30cc4fa2899, fk-352d7b3fcad4 | spawn | 2 | — |

No key appears in two streams (the synthesis's stream partition, verified by the digest). Keys outside every
stream: the repo/h10 `slug:deferred-task-flip-conditions` prior (owned by the deferral-records PRD's filed
tasks, contract §7), and the mechanical findings filed in Phase 5 (next section), which appear here only by
task id.

## Mechanical findings already filed (Phase 5)

67 mechanical findings were filed by the run itself after the contract §8 dedup protocol (55 new
tasks 6587–6646, 5 curator-combined into 3245/4469/5462/5747/6037, 7 matched to live tasks 3150/6213/
6520/3416/5287/5924/3327). A stream whose remedy subsumes one of those tasks declares a dependency on it or
cancels it with `x_acceptance_reason` naming the subsuming leaf; it never files a second task for the key.
Full table: the report's `## Mechanical filings`.

## Seam ownership (G4 — authoritative)

One owner per shared artefact; consumers import, never redefine. The owner's stream lands the seam as its
first leaf where a consumer stream depends on it.

| Seam / artefact | Owner | Consumers (do NOT redefine) |
|---|---|---|
| `shared.git_async.run_git` as the factory-wide git runner | S5 | verify (S4), merge_lane (S6/S7), escalation server (S3), fused-memory probes (S10) |
| `GitOpsCallbacks` / constructor-injected GitOps ports; `LaneLifecycle` durable lane record; `AdvanceOutcome.detail` | S5 | harness (S8) constructs; merge lane (S6) consumes `AdvanceOutcome.detail`; pool and harness go through `LaneLifecycle` |
| `WarmLaneUnavailable` members carrying exception + `BlockDisposition` | S5 | workflow_types rows derive (S9) |
| `VerifyContext`, typed `VerifyResult` (`FailureCategory` enum, `CheckRun`) | S4 | merge lane (S6/S7), workflow (S9), harness (S8), cli build it; `verify_categories` owns the vocabulary |
| `attempt-*` verify artefact grammar helper | S4 | `scripts/verify_budget_census` (S25) parses through it |
| `MergeOutcome.kind`, `GroupMergeRequest.train_kind`, `ItemLifecycle` as operative state, `MainHealthAutoHeal` collaborator, landed-row journal injected by harness | S6 | workflow (S9) and unblock_types consume; harness (S8) constructs the journal |
| `GateMeasurement` | S7 | worker (S6) injects the fail-open policy |
| `MergeLaneQueries` port (definition, `orchestrator/merge_lane/queries.py`) | S3 defines (α 9152), S6 implements natively (μ 9201, with the one total mapping onto the public `MergeState`); no interim adapter anywhere (R25, R27) | the merge tools registered from `orchestrator/mcp/` (S3 ε1 9188, ε2 9193) consume |
| `escalation/filing.py::file(record, FilingPolicy)`; `EscalationPort.file(record, dedupe)` widening | S1 | merge lane, scheduler port, `TaskWorkflow._file_escalation`, recon registry (S12) choose a policy only |
| `escalation/models.py` typed vocabulary, `cascade_parent_id`, `ACTION_EFFECTS` with `record_disposition`, headers + sentinel prefixes, `parse_caller_identity` | S2 | dashboard (S22), harness (S8), scripts/sitting (S25), S1/S3 import |
| `queue.py::_mutate_pending`, per-task index; ladder tools registered from an escalation module; merge/ops tools registered from `orchestrator/mcp/`; the `WorkflowSlots` Protocol (S3 δ 9167, `orchestrator/mcp/ops_tools.py`) | S3 | server never calls `_rewrite`; S8's `SlotRegistry` reaches the server through S3's port and implements `WorkflowSlots` (S8 keeps no second definition) |
| `SlotRegistry`; `SchedulerCallbacks` as the only Scheduler hook mechanism; `ManagedServices` teardown order; `PauseClass` | S8 | escalation server (via S3), scheduler (S17) |
| `TaskGroundTruth.derive_truth` with condition rows as the one recovery classifier | S8 (task_ground_truth) | both harness sweeps, scheduler stranded phase (S17) |
| `stranded_verified_green.py` merge-done classification; typed `BlockReason`; Plan model with `PlanState` in `artifacts.py`; `IterationEntry` | S9 | harness (S8) consumes the classification; dry_run_unblock consumes `BlockReason`; plan_tools, harness, evals/judge read the Plan model |
| `shared.task_metadata` wire contract (`metadata_mode`, `text_mode`, static slice registry, `declared_files`), `MetadataWriteFailed`, `PrivilegedAuditWriter`, candidate-key scope SQL | S10 | fused-memory backend and interceptor, orchestrator scheduler (S17) and workflow (S9) import; `SchedulerFacade`/`FakeScheduler` parity in S10 |
| `curator_types.py`; `resolve_ticket` status vocabulary (non-terminal `pending`) | S11 | roles.py quotes the vocabulary |
| `RunContext`/`StageId`; the one recon run driver with `RunKind` | S12 | stats_verifier, judge, recon_report; `BacklogIterator` calls the driver |
| flag identity leaf; `RunFindingIndex`; `FindingCategory` enum; `shared.task_statuses`-derived lexicon | S13 | server/recon_report, scope_freshness import the leaf; cli_stage_runner schema derives |
| `shared/project_layout.py`, `project_id.py`, `merge_subject.py`, `scheduler_overrides.py`, `SchedulerStateSnapshot`, claimant TTL constant, shared streak primitive, `shared.mcp_post` as the only MCP client | S14 | dashboard, fused-memory, orchestrator, scripts, legibility import; S12/S13/S17/S19/S24/S25 depend |
| `ToolContext`; dispatch wrapper as the one scope/markup gate; shutdown tuple; fused-memory read model for dashboard panels | S15 | tool family modules import downward; config/schema.py mirror derives; dashboard (S22) consumes the read model |
| shared routing resolver + allowlist; shared hot-reload engine; `RoleName`/`Backend` enums; field defaults as the one home | S16 | orchestrator and fused-memory config import; cli_invoke/invocation_outcome (S20) import the enums |
| `get_state_snapshot` typed test seam; `delivered_checks.run` as the one evaluator | S17 | landing_evidence (S6) and scheduler call the evaluator |
| `evals/results.py ResultStore`; `ReviewPayload` normaliser | S18 | scripts/_eval_results_io, backfill import |
| `shared.config_dir.pid_alive`; `fused_memory.recon_state`; shared sandbox module; shared session package | S19 | the five other `_pid_alive` copies become imports; S12/S13 populate recon_state; invoke and sandbox_guard import the sandbox module; cockpit and scripts import the session package |
| `classify_invocation` as the one classifier; `slot.observe -> Decision`; `CliSubtype`; transcript leaf; `shared.capability_manifest` public surface | S20 | usage_gate and cli_invoke consume; dry_run_unblock, routing, steward, path_scope_adjudicator import `CliSubtype`; S25 files `sidecar_for` as a dependency on S20 |
| legibility detector module; codebook merge result | S21 | sampling, census_window consume |
| label→root resolver (active_tasks); one fan-out classification + `TTLCache` owner; the cross-package `runs.db` events read model (S8-3 follow-up, R36) | S22 | all dashboard readers; fused-memory, the client feed, the quality series and scripts for the events read model |
| `MetricGroup` table; `sampler.store` schema public | S23 | dashboard load.py and calibration import |
| shared fleet-coordination leaf (lease keyed by tier, clock, ages); `scripts/_systemd.py` host reader | S24 | service_restart, watchdogs, df_pytest_isolation, bash via `python3 -c` |
| `scripts/_checkout.py` bootstrap; `run_parity` driver | S25 | every script |
| `shared.route_rejection` (structured routing rejections {layer, reason, model}) | S25 (λ 6766) | S16 consumes when it moves the resolver; S9 one-line emit-site change |
| `GraphAccess` protocol; queued-write value types; `MemoryService` as the only graph mutator; `EpisodeIdentityReconciler`; identity lock as public context manager | S26 | tests and cross_graph_move use the public provider; durable_queue imports the value types; sweeps call the read surface and `delete_entity` |

## Design invariants promoted

Eight invariants were promoted in checkable form (R22, R1). Each owning stream performs the four-site
lockstep edit `CONTRIBUTING.md` §6 describes and mints the INV id there; this document carries pointers
only (INV-9): filing entry point (S1), escalation literal ban (S2), fused_memory subpackage DAG (S19),
static metadata-slice registry (S10), config default single home (S16), `GateMeasurement` unmeasurable
(S7), sampler metric table (S23), lane timing-budget ordering validator (S6). The INV ids are appended to
the FILED table rows of the landing leaves.

Owners after wave 2 (filed 2026-11-11; landing order R51, every check anchored on the slug, never the
number):

| Invariant (slug) | INV id at landing | Stream | Landing leaf | Status 2026-11-11 |
|---|---|---|---|---|
| `lane-budgets-exceed-verify-timeouts` (lane timing-budget ordering validator; R1, R22 item 8) | INV-15 (first) | S6 | α 9140 (the lockstep edit folded into α) | pending |
| `metadata-slices-one-table` (static metadata-slice registry; R3, R22 item 4) | INV-16 (after 9140) | S10 | θ 9242 | pending |
| `merge-gate-unmeasured-typed` (`GateMeasurement` unmeasurable; R11, R22 item 6) | INV-17 (after 9242) | S7 | ζ 9154 | pending |
| `sampler-vocabulary-one-table` (sampler metric table; R22 item 7) | minted | S23 | β 6773 | done |

S3, S4, S5, S8, S9 and S26 promote no invariant. Still unowned by a filed task: the filing entry point
(S1), the escalation literal ban (S2), the fused_memory subpackage DAG (S19) and the config default single
home (S16) — all wave 3.

## Resolved decisions (do not relitigate)


## R1 — S6 Merge-lane lifecycle, outcome types and provenance (2026-10-10)
Ruling: option 1, class decomposition, sequenced typed-facts first. First leaves: the load-time timing-budget validator, `AdvanceOutcome.detail` replacing the GitOps `_last_*` side channel, the landed-row journal injected by the harness (no `MergeProvenance.bind`), `MergeOutcome.kind` / `GroupMergeRequest.train_kind` enums replacing prefix routing, one `MainHealthAutoHeal` collaborator. Then `ItemLifecycle` as the operative state with one `_retire(item, outcome)`, the collaborator decomposition of `SpeculativeMergeWorker` (lifecycle, admission, dispatch, quarantine, snapshot), and retirement of the 14 alias shims.
Options considered: (1) decompose first; (2) move the three symptom-free partitions now, class stays whole; (3) typed facts only, accept the h13/h14 residue as standing.
Consequence that decided it: option 2 pays the decomposition cost later without reducing it, and the live hazards (silent train halt, budget ordering) rest on the typed-facts leaves, which option 1 lands first anyway.
Invariant: PROMOTE — "a lane timing budget exceeds every per-command verify timeout the config can authorise", enforced by a config-load validator with a fixture (a yaml raising `verify_cold_command_timeout_secs` above the budget fails load). New INV via the CONTRIBUTING.md §6 four-site lockstep edit; owned by the S6 stream.
Keys: fk-313f404f3cc6, fk-a2984bf98f85, fk-e61eed486dbc, fk-80a790e6f793, fk-c55b12f4b66d, fk-e8a03e4ec067, fk-ba531799522b, fk-1432a52e4753, fk-769c3bb58ce1, fk-153195971a14, fk-e521254a5b71, fk-e7deb8bcfa95, fk-e768d50f8363, fk-83625b943aa5, fk-201d70aaf409, fk-a0f6af37f0d6, fk-59b3df6f52a3, fk-2c9b6486d554, fk-e501862ffe2c, fk-d9e6c03dc259, fk-d41caf6b6331, fk-4b76c9695674, fk-7f2bce2d482c, fk-fd91b14b3ead, fk-75fc1b96098d.

## R2 — S9 Workflow phase spine and recovery ledger (2026-10-10)
Ruling: option 1, a typed Plan model in `artifacts.py` with `PlanState` as a derived property; `mcp/plan_tools.py` and `stamp_plan_provenance` write through it with finalization as a precondition. Rest of the S9 remedy as proposed: `IterationEntry` discriminated union with `is_work` decided at the writer; the phase loop returns a typed `BlockReason` and `_mark_blocked` takes it; `StewardOutcome` returned on the channel; `_invoke` split into resume-arming / invocation / accounting with one route resolver (`routing_dispatch.resolve_and_record_route`); protocol-typed seams with conforming doubles; merge-done classification and stranded_merge_failed filing move to `stranded_verified_green.py` for workflow and harness alike.
Options considered: (1) typed Plan model; (2) raw dicts plus a `plan_state(plan)` helper; (3) accept the plan lifecycle as standing.
Consequence that decided it: option 2 leaves the writer-side marker desync that produced fk-37190b541196; the coupling option 1 costs is to a schema the plan-tool MCP process already writes by hand.
G4 seam notes: `BlockReason` (workflow_types) is S9's artefact; the task-write success signal is S10's; recovery sweeps (S8) consume `stranded_verified_green.py`'s classification, never a second copy.
Keys: fk-37190b541196, fk-9973521f4f59, fk-108cdd00d122 (weakened), fk-f37f7e253cd5, fk-2c0594928a18, fk-ee708be4436f, fk-4f7a933c0dd4, fk-a152341e38ed, fk-44bc973b1cc4, fk-d2ba7601c130, fk-bab6058b23c8, fk-e39550272df9, fk-9eb149f0706d, fk-a503ccb37b4b, fk-a71254176a81, fk-b8efaa973162, fk-7d410939a337, fk-67b5971ed72d, fk-e43472e7d5d7, fk-5ea52a0f89ad.

## R3 — S10 Task-write contract: metadata modes, success signal and registry (2026-10-10)
Ruling: option 1, `Scheduler.update_task` raises a typed `MetadataWriteFailed` carrying the rejection envelope; the 13 documented except arms in workflow.py become live as written; S9's `BlockReason` leaf is where the raise becomes a block. Rest of the S10 remedy as proposed: one wire contract in `shared.task_metadata` (`metadata_mode` Literal required with metadata, `text_mode` for details/prompt, resolved once in the backend, deleting the client-side resolver); every metadata slice registered statically in `shared.task_metadata` (slice models in leaf modules, no import-side-effect registration); one `declared_files(metadata)` reader feeding candidate_key, CandidateTask and lock derivation; one `raw_metadata_dict` parse per request; the status FSM enforced in the backend as well as the interceptor; a `PrivilegedAuditWriter` protocol replacing the three `type: ignore[attr-defined]` calls; one candidate-key scope SQL constant and collision helper; curator combine writes the file union into `metadata.files` via the sanctioned sanitizer; the 308-line blessed-key comment block rehomed (docs/task-authoring.md §8 is the vocabulary home; census figures leave the code).
Options considered: (1) raise; (2) keep bool through one `_persist_metadata` helper.
Consequence that decided it: the suite (17 files) and the code already encode the raising contract, and the three money-burning-loop guards only function under it; a bool helper leaves silent False possible for any bypassing site.
Invariant: PROMOTE — "parse_metadata's registered slice set is a property of shared.task_metadata alone: identical in a fresh interpreter whatever first-party module was imported first", fixture spawns `python -I` per entry module (shared.task_metadata, fused_memory.backends.sqlite_task_backend, orchestrator.scheduler, orchestrator.cli, dashboard.app) and compares the registry. New INV via the §6 four-site edit; owned by S10.
Seam ownership: shared.task_metadata (S10 owns; fused-memory backend and orchestrator scheduler import); SchedulerFacade/FakeScheduler parity (S10); candidate_key scope SQL (sqlite_task_backend, S10).
Keys: fk-ff5234ac1013, fk-116481623cfb, fk-de964b86bdd8, fk-f6224c565a9b, fk-1f8a4c69798f, fk-fa3e3d765f50, fk-f528087f44c7, fk-941b5369a3d7, fk-d80c1c44bca1, fk-d4d61005457b, fk-260352fa900a, fk-cc04e61d23ec, fk-06ab2d3e9148, fk-c41aa02e0779, fk-85e1293e0aa4, fk-1eb04ceac9e8, fk-e3e99447e11e, fk-de3df95984a5, fk-fa355c2c0498, fk-1695d6b02400.

## R4 — S4 Verify run context and runner contract (2026-10-10)
Ruling: option 1, infra-transient is data on a typed `VerifyResult` (`FailureCategory` enum incl. sentinels, `CheckRun` with `ToolKind`, `timed_out`/`infra_transient` as fields); callers branch; `VerifyInfraError` and the xdist-crash raise are retired, the merge lane's stall-on-raise becomes an explicit branch. Rest of the S4 remedy as proposed: frozen `VerifyContext(resource_class, worktree_provenance, verdict_authority, retry_owner, module_registry)` with constructors `merge_gate()/main_probe()/task()/background()` replacing `role`/`is_merge_verify`/`allow_cold_cache` and the `OrchestratorConfig._module_configs` private channel; one no-source-diff resolver for both branches; one `derive_verify_plan` engine; one isolated-rerun policy table; one `attempt-*` artefact-name helper (scripts/verify_budget_census parses through it); `verify.py` split per the seat's measured partition after `_worktree_reader` and `_emit_trivial_pass_escalated` move down; offline-lane legs as `LaneCommand` values under one spawn helper; a schema version on the dispatcher-to-remote wire.
Options considered: (1) data on VerifyResult; (2) `VerifyContext.retry_owner`, raise only when set.
Consequence that decided it: one contract across the seven entry points removes the "which callers can absorb this" class entirely; option 2 keeps two outcome channels flipped by a context field the next agent must know about.
Seam ownership: `VerifyContext` (S4 owns; merge lane S6/S7, workflow S9, harness S8, cli build it); `FailureCategory`/sentinel vocabulary (`verify_categories`, S4); `attempt-*` grammar helper (S4 owns; S25 consumes).
Keys: fk-0bb03714372a, fk-5037fafc755c (weakened), fk-0f9cace0cd00, fk-fac2cb6af858, fk-d99f146ff901, fk-0c5e73f13485, fk-8dbae7d39adc, fk-e906c4e914c2, fk-5fa21a4b85cb, fk-baa454b81f6f, fk-3756ad2f3357, fk-f994995001e5, fk-caa8f680e1c2, fk-3b67e8761674, fk-09ad1a19f587, fk-a2fb58789c14, fk-c88f7d3ae54a, fk-37d63bf83608, fk-cb29fac47dba (weakened), fk-126a75a57ca5, fk-73b03674a8e4, fk-a635d54da4ab, fk-3e806baf4c6e, fk-591206484ea5 (weakened).

## R5 — S5 GitOps ports, query contract and the git runner (2026-10-10)
Ruling: option 1, `WarmLaneUnavailable` members carry their `WarmLaneRequeue` class and `BlockDisposition`; one raise site in `create_worktree`; the `workflow_types` warm-lane rows derive from the enum (deleting the 9-arm if-chain, the FAULT literal class list and the `_STEAL_RETRYABLE` placeholder binding). Rest of the S5 remedy as proposed: `shared.git_async.run_git` is the factory-wide git runner (the 6 src importers of `git_ops._run` and fused-memory's 3 hand-rolled done-provenance probes repointed; git_ops keeps only the `WorktreeMissing` adaptation); GitOps gets constructor-injected ports (WarmLaneScripts, base-health probe, limits dataclass) and a `GitOpsCallbacks` bundle, replacing the 4 post-construction attribute installs and the 128 private patch sites; one `GitQueryError` contract for the 11 rc-dropping queries; one typed diff reader behind the 4 parsers; one `_move_main` primitive under `advance_main`/`recover_red_main`; one lane-lease context manager shared with `cli.verify_merge` and `cancel_verify`; `LaneLifecycle` as the single public owner of durable lane records (Harness receives it, never `git_ops._lane_lifecycle`); `AdvanceOutcome.detail` is S5's type (git_ops owns), consumed by S6 per R1.
Options considered: (1) enum carries exception + disposition; (2) drop the enum, raise typed subclasses.
Consequence that decided it: option 1 collapses three encodings to one table with one raise site and derives the disposition rows (the same shape as S9's BlockReason); option 2 leaves two parallel hierarchies kept in step by hand.
Seam ownership: `shared.git_async.run_git` (shared, S14/S5; orchestrator, escalation, fused-memory import); `LaneLifecycle` durable record (S5; pool and harness go through it); `GitOpsCallbacks`/ports (S5 owns; harness S8 constructs); `AdvanceOutcome.detail` (S5 owns; S6 consumes).
Keys: fk-b1e3368f99a1, fk-6f7520ff1987, fk-e2ac53b0d72b, fk-d727d82341b3, fk-d176c9ae1cc7, fk-7a3ee979b48a (weakened), fk-78b757799cab, fk-a7cefe40ef4c, fk-a7d433cf8308, fk-ad8c0528c5e1, fk-eb7b9f65f412, fk-3ee93d431321, fk-856541951d65, fk-c5d5769d68c6, fk-718c82fcc1a2, fk-739af676767e, fk-c00c5f01be4c, fk-d73aeaebb792, fk-3a4cc6298c02, fk-31e9856ed662, fk-a10d37257808, fk-54eb9f7cc0be, fk-20135ef54f75, fk-c04bf2045ad3.

## R6 — S8 Harness composition root and slot registry (2026-10-10)
Ruling: option 1, both recovery sweeps consume `TaskGroundTruth.derive_truth` verbatim; the four applier rewrites in `_reconcile_one_stranded` (LEAVE→MARK_DONE, LEAVE→RE_FILE, etc.) become `_RECOVERY` table rows with condition columns, and the sweeps shrink to dispatch. Rest of the S8 remedy as proposed: Harness as a pure composition root with a `SlotRegistry` (one frozen per-slot record registered at dispatch, released in one done-callback, deterministic slots included) read by `is_workflow_active`, the resolve path, the orphan reaper and the escalation server's `release_workflow` through S3's port; every hook passed at construction (`SchedulerCallbacks` absorbs the two private Scheduler gates, `GitOpsCallbacks` from S5, required non-Optional safety gates; finishes task 2235's migration); `usage_gate`/`cost_store`/MCP registered as `ManagedServices` first so `stop_all` tears them down last (deleting the hand-ordered finally); sandbox backend and MCP session passed with the invocation (deleting `sandbox_dispatch.set_backend`, both restore fixtures, `mcp_lifecycle._session` and the `rstrip('/mcp')` misparse); `PauseClass` enum persisted beside the reason (deleting 4 `ewa_trip_` parses); the session-resume storm detector extracted so `storm_window_bound` stops reaching back; `EventStore.emit` batched on one connection (INV-8); sweeps and filers as classes receiving their collaborators, retiring the 37 `Harness.__new__` fixtures' tolerance reads; non-claude backends write per-invocation config outside the worktree's staged tree.
Options considered: (1) verbatim with condition rows; (2) base verdict plus sweep-side policy.
Consequence that decided it: TaskGroundTruth was built as the single classifier (TG-2) and has one consumer that overrides it four times; option 2 ratifies the override and leaves three places answering what happens to a stranded task.
Seam ownership: `SlotRegistry` (S8 owns; S3's port exposes it to the escalation server); `SchedulerCallbacks`/`GitOpsCallbacks` as the only hook mechanism (S8 wires; S5 owns the GitOps bundle); `TaskGroundTruth.derive_truth` (task_ground_truth owns; both sweeps and the scheduler phase consume).
Keys: fk-c1ac3df8e44d, fk-d8f08696dd66 (weakened), fk-fc3c4cba6a0f (weakened), fk-b01dac97e6e9, fk-d0c02dc767dd, fk-1e13b183c961, fk-b1c6965066ed, fk-d470c7828682, fk-17b789fe3ce6, fk-e6a18c90d2fe, fk-bf4197a9a7d9, fk-5cad48d24b6f, fk-5d0a66a2d32e, fk-a1085af3a290, fk-f1a7f57fa58e, fk-a97501641592, fk-538940c8d347, fk-58d1f3489c98, fk-9854bd9a928f.

## R7 — S26 Memory service write path and graph client (2026-10-10)
Ruling: option 1, an `EpisodeIdentityReconciler` owned by `MemoryService` holds the nine identity passes and their counters behind a public `run()` (the 241 private test references move onto it). Rest of the S26 remedy as proposed: typed queued-write values per operation in a leaf beside `durable_queue` with `to_json`/`from_json` and an operation enum (deleting the 4 untyped producer dicts, the pop-then-read keys and the silent fall-through to a Graphiti write); `MemoryService` constructed with its always-present collaborators and queue workers started by an explicit `start()` after wiring (deleting the 24 write-journal None-guards and the startup drain window); backends private with a narrow read surface for sweeps and every graph mutation through the journaling layer (9 reach-through sites repointed; the degenerate-node sweep's delete goes through `delete_entity`); a `GraphAccess` protocol injected into `GraphitiBackend` with bounded-timeout egress (fakes via the public constructor; the 220 `_driver._get_graph` stubs and 69 `_graph_for` patches go); one `_journaled_op` context manager (187 lines of finally boilerplate and 9 `success = True` go); `search` split into fan-out / fuse-and-filter / journal with the topic-anchor pin as a pure function in `topic_anchor`; one `_RELOCATED_EDGE_PROPS` list (or `SET new += properties(old)`) replacing the 4 SET-list copies and the test tuple; the identity lock as a public context manager used by merge/rename/delete/reassign and cross_graph_move; `add_memory`'s response stops reporting Graphiti as written when only queued.
Options considered: (1) EpisodeIdentityReconciler; (2) keep the passes as methods, extract formatters and codec only.
Consequence that decided it: the identity chain is the one partition the h14 measurement found separates cleanly, and moving the private test touches onto a public run() is the Tests-stance fix itself, not a by-product; option 2 leaves the file above every alarm with the reach intact.
Seam ownership: `GraphAccess` protocol (backends own; tests and cross_graph_move use the public provider); queued-write value types (services own; durable_queue imports); `MemoryService` as the only graph mutator (sweeps call its read surface and `delete_entity`).
Keys: fk-1215aacfa626, fk-017442f273d2, fk-9430031679c4, fk-eb0d5fea9a52, fk-265caf7191a5, fk-d37d63bf0d11, fk-6e50b22357f2, fk-abc3487a3528, fk-9804ecc4567d, fk-62af616b8470, fk-90f6fbb5daa0, fk-da5b47d0493a, fk-50cc281018fb, fk-343c4a42be6b, fk-bdc0427c8971, fk-41eaf65cc4d0, fk-b30cc4fa2899, fk-352d7b3fcad4.

## R8 — S1 One escalation filing entry point (2026-10-10)
Ruling: option 1, a queue-layer filing function `escalation/filing.py::file(record, policy: FilingPolicy)` with injected status/claimant lookups, owning the five chokepoint gates now in `server._chokepoint_or_submit` (eval-lane containment, claimant stamp, born-at-L2, terminal auto-resolve, dedupe) plus the to_thread hop; `EscalationQueue.submit` becomes its private write step; MCP tools and in-process filers (merge lane via a widened `EscalationPort.file(record, dedupe)`, one injected scheduler port replacing 7 callbacks + 3 inline sites, `TaskWorkflow._file_escalation`, recon's typed category registry) all go through it; the 95 submit sites migrate in this stream.
Options considered: (1) queue-layer function; (2) keep the server chokepoint, in-process filers call the MCP tool over HTTP.
Consequence that decided it: the harness hosts the server, so under option 2 the harness's own alarms would depend on a server the harness may be restarting; option 1 writes a new ladder rule once.
Invariant (synthesis candidate 1): "every escalation record in the queue directory was written through the one filing entry point" — fixture monkeypatches `EscalationQueue.submit` to record callers; static form greps `Escalation(` constructions outside filing.py. Promoted with S2's (R9) as the pair that S1/S2 own.
Seam ownership: `escalation/filing.py` (S1); `FilingPolicy` + the 29 dunder sentinel task_id literals (one registry in `escalation/models.py`, S2 owns the models, S1 the policy); `merge_lane/ports.py EscalationPort` widening (S1 edits; S6/S7 consume); recon category registry (S12 consumes S1's `FilingPolicy`).
Keys: fk-7d1f3a0d12ca, fk-333277fb4bd7, fk-0d31f1114911, fk-32903f2e64e0, fk-2e0a4419290b, fk-cd446ff3da74, fk-9c72e4399b4e, fk-1bf4bfa8be04, fk-b7b0be25c6b4, fk-0ee68c0948ab, fk-1b573c2428c2.

## R9 — S2 Typed escalation record facts (2026-10-10); resolves contradiction 1
Ruling: option 1, a persisted `Escalation.cascade_parent_id` (schema change with archive migration; old records default to None), replacing the `l2-cascade:<id>` prefix inside `resolved_by` and its two harness parses; `resolved_by` becomes attribution only. Rest of S2 as proposed: `escalation/models.py` as the typed vocabulary (Status/Severity/Category/ResolveAction StrEnums with OPEN/TERMINAL subsets; `Escalation` gains `filing_site` and `failure_signature`); `ACTION_EFFECTS` gains `record_disposition` so `resolve_issue` branches on the effect and `RESOLVE_ACTIONS`/`_DISMISS_ACTIONS` derive from it; the resolve/park callback receives a frozen event; one `parse_caller_identity(headers)` for the four header-reading tools; the dead `CATEGORIES` enum and the 12 dashboard status literals go; roles.py's ladder prose points at the table instead of restating it.
Options considered: (1) persisted field; (2) frozen callback event only.
Consequence that decided it: the deferral-records PRD (plans/deferral-flip-condition-prd.md, decomposed) already requires a holder/notice fact that survives restarts, and `classify_resolver_tier` needs a persisted marker either way, so option 2 reintroduces a second encoding.
Invariant: PROMOTE — "no module outside escalation/ compares an escalation status, action or category to a string literal; the vocabulary is imported from escalation.models" — AST scan over dashboard/src, orchestrator/src, fused-memory/src, scripts for Compare nodes against the vocabulary; the checkable form of INV-5 for this vocabulary. New INV via the §6 four-site edit; owned by S2.
Seam ownership: `escalation/models.py` vocabulary (S2 owns; dashboard S22, harness S8, scripts S25 import); `ACTION_EFFECTS` as the sole action table (S2 owns; harness and scripts/sitting consume).
Keys: fk-8c48269c7363, fk-5332d25b1702, fk-e3b7459c681d, fk-d0e1094c99cb, fk-bbb0a7b3c8a8, fk-60b7c33ea0ad, fk-638cca2f65bb, fk-479375564c2f, fk-755f786c32be, fk-48a76ee9f746, fk-ece88b7661b7, fk-9c7ebc2b50b2, fk-a65731424439, fk-89388166d3ca.

## R10 — S3 Escalation server and queue boundary (2026-10-10); resolves contradiction 2
Ruling: option 1, `orchestrator/mcp/` registers the merge/ops MCP tools onto the FastMCP instance escalation hands back, fed by a typed `MergeLaneQueries` port (`live_request_state`, `owns_merge_worktree`, `terminal_state`) instead of `getattr(harness, '_merge_worker')` plus `snapshot()` dict parsing; escalation keeps zero orchestrator imports (the 9 function-local reach-backs go). Rest of S3 as proposed: `create_server` split along the seat's partition (ladder tools registered from an escalation module taking `(mcp, queue, dedupe_config, lookups)`); `queue.py` gets one locked pending-only `_mutate_pending(id, fn)` under every per-record mutator and `resolve_issue`'s rewrite (7 RMW copies go), a per-task index for `get_by_task`, amendment policy and scan helpers moved to leaf modules with importers repointed (no shim); `_WATCHER_ESCALATION_HEADERS` joins the sentinel prefixes in models; `get_task_escalation_history` made archive-inclusive as its name promises.
Options considered: (1) orchestrator/mcp registers; (2) keep in escalation behind a typed port.
Consequence that decided it: only option 1 makes the layering rule "escalation never imports orchestrator" checkable and keeps the ladder server reusable by fused-memory and scripts; option 2 grows the port per tool and makes escalation the standing home of orchestrator ops.
Seam ownership: `MergeLaneQueries` port (orchestrator S6 owns the implementation, S3 the port definition; escalation server consumes); `escalation/models.py` headers + sentinel prefixes (S2 shares); `queue.py _mutate_pending` (S3; server never calls `_rewrite`).
Keys: fk-a8d3003e9d83, fk-e903cc072b15, fk-43f431f697a3, fk-609a1832903b, fk-4a0626fc033d, fk-5a9e713f4055, fk-e5d81cc1f1d4, fk-07be43f313bc, fk-f8e9e0da3786, fk-66746efddd4d, fk-bc64099a1cd0, fk-ce2dc99e0305, fk-43bad4e724da, fk-c8c3406fd7f4, fk-0de72a91f89a.

## R11 — S7 Merge-lane gates, placement mechanisms and target adapters (2026-10-10)
Ruling: option 1, delete the variable-depth probe, deep-chain and auto-chain placement mechanisms (~1,450 lines, every project config has them off) and keep one adjacent placement; `GateMeasurement` (measured(value) | unmeasurable(reason, rc, stderr)) returned by every git-probing gate, fail-open/closed decided in one place in the worker, every unmeasurable result emitting `merge_gate_unmeasured` under a storm counter (replacing `_OVERLAP_GIT_ERROR_SENTINEL`, `_RENAME_PROBE_UNMEASURABLE` and the 12 fail-open pass-value returns); reify retry narrowing behind `build_retry_env` in a target-verify adapter module with a contract pointer (~730 reify-specific lines leave the worker); resource-audit violations as records fingerprinted on (kind, subject).
Options considered: (1) delete all three; (2) keep probe and deep chain behind a PlacementPolicy port, delete only auto-chain.
Consequence that decided it: the lane is being decomposed under R1 and carrying dead placement paths through that decomposition doubles its cost; re-enabling is a re-implementation against a cleaner lane.
Invariant: PROMOTE (synthesis candidate 6) — "a merge gate that cannot measure returns GateMeasurement.unmeasurable and emits merge_gate_unmeasured; no gate returns its pass value from an rc != 0 or except branch" — AST over merge_lane/gates.py: every Return inside an except or rc != 0 branch of a `_check_*` function returns a GateMeasurement with measured=False; the checkable INV-11 form for the lane. Owned by S7.
Seam ownership: `GateMeasurement` (gates.py, S7 owns; worker S6 injects the fail-open policy); target-verify adapter module (S7; reify contract pointer).
Keys: fk-88e3c228b649, fk-689227abfab1, fk-7d1580bb3f03, fk-dd411ada7c29, fk-5d4406afeade.

## R12 — S12 Reconciliation run lifecycle and stage interface (2026-10-10)
Ruling: option 1, one run driver (start_run → stages → pre-S3 flush → complete_run → verify → judge → except ladder) parameterised by `RunKind(full | remediation | backlog_chunk)` and scope/events, with scope-freshness as a pre-step, used by `run_full_cycle`, `_run_remediation_pass` and `BacklogIterator`; the CancelledError/resume asymmetry between the two copies is decided as one rule in the stream's PRD. Rest of S12 as proposed: `ReconciliationRun` typed (StageId-keyed `StageReport` map, explicit error/resume attributes, journal as sole rehydrator); `StageId` in the run context rather than a prompt-echoed agent_id string; `assemble_payload(…, run: RunContext) -> PayloadResult(text, stats)` removing the 10 instance stashes and the 11 "set by harness" attributes; stages constructed with their collaborators; the two journal-recomputed streaks as declarations over S14's shared counter; `targeted.py` classified uniformly with an event per warning path; the fused-memory URL derived from server config, not a hard-coded `.mcp.json` default.
Options considered: (1) RunKind on the one driver; (2) full cycle only with pre/post hooks for remediation.
Consequence that decided it: the two copies differ in scope and in one asymmetry that is itself a finding; a mode value forces the asymmetry to be decided, hooks would preserve it and grow into a second lifecycle.
Seam ownership: `RunContext`/`StageId` (reconciliation owns; stats_verifier, judge, recon_report consume); the one run driver (harness.py owns; BacklogIterator calls it); shared consecutive-streak primitive (S14 owns).
Keys: fk-8621784d79d3, fk-2571f7726184, fk-377a7036ac7f, fk-f23c6e137ce5, fk-0cce86f5dbfe, fk-a3598aec7dca, fk-d2c7f1b902ac, fk-63b65973bd33, fk-37356472614a, fk-90e488a174b7, fk-a88506a1e327, fk-0485729893fe, fk-16baf39b1db6, fk-147a9951edc7, fk-10c8c43d9e89, fk-20cb93d17ff3, fk-9850339b610c, fk-be8ed2cd9ce9, fk-05443fb78da6, fk-2bb6c913d747, fk-875cca636438.

## R13 — S13 Recon sweeps, flag identity and recon-report state (2026-10-10)
Ruling: option 1, suppression matches when any component of a composite task_id matches a suppression row; the composite is parsed once into a typed subject that keeps every component so an over-suppression is visible in the ledger. Rest of S13 as proposed: a public flag-identity leaf (fingerprint, normalizer, signature, `FlagTypeFamily`, UUID extraction, composite task-id parser) stating the persisted-format contract once, imported by `server/recon_report` and `scope_freshness` (the two private `_content_fingerprint` imports go); every flag_type vocabulary a `FlagTypeFamily` with `log_drift` (4 of 5 matching mechanisms and 3 hand-rolled drift logs go); status lexicons and terminal sets derived from `shared.task_statuses` (3 lexicons + 3 terminal sets go); one Graphiti edge-sweep primitive with the agent id from `StageId`; `task_filter` split into `recon_prose_grammar.py` and the tree renderer; marker pools move from Mem0 to the recon ledger; `ReconReportState`'s six run maps become a `RunFindingIndex`; `FindingCategory` StrEnum parsed at `add_finding` and generating the `cli_stage_runner` schema enum.
Options considered: (1) any component matches; (2) exact composite match.
Consequence that decided it: operators cannot know Stage 1's composite in advance, so exact matching keeps failing on composites and invites a fifth parser.
Seam ownership: flag identity leaf (reconciliation owns; recon_report and scope_freshness import); `shared.task_statuses`-derived lexicon (task_filter builds it once; S14 owns the source); `RunFindingIndex` (recon_report owns).
Keys: fk-51ec1e4010af, fk-917fdf32a542, fk-5d00812d5256, fk-d9bfbf342b5b, fk-65aeb80b161d, fk-e64ee986eeed, fk-9f66b6f4420c, fk-b1cda781c60e, fk-9165ae3a1e72, fk-90f374fcce21, fk-fa1fd33c7a9e, fk-9b76dd8a1cdb, fk-cdb8c20a3d0f, fk-2868338d3cf3, fk-46514b4ef0cc, fk-fbbf580e4e1d, fk-4c8d19c64d99, fk-e30ecd1f36b8, fk-0791e9817b84, fk-7c13df0b988c.

## R14 — S15 fused-memory MCP server boundary (2026-10-10)
Ruling: option 1, project scope declared per tool (decorator) and parsed once by the dispatch wrapper that already hosts `markup_guard`; a missing declaration is a registration error. Rest of S15 as proposed: a `ToolContext` (interceptor bound once as real or null object, write journal, triage counter, scope parser, read-journal decorator) that per-family registration modules import downward so the 9,386-line `create_mcp_server` closure becomes a composition; `add_memory`/`submit_task` guard chains as ordered gate lists; one `WRITE_TIME_CONTROL_KEYS` registry stripped in `_extract_causation`; `AddMemoryResponse` carries per-store outcomes; `main.py`'s shutdown as a declared (name, closer, timeout) tuple that the budget test and the config/schema mirror derive from; collaborators via constructors with one `TaskInterceptor` builder (12 post-construction wirings go).
Options considered: (1) declarative scope via the wrapper; (2) explicit `parse_scope` first line per tool.
Consequence that decided it: the finding is an omission class (7 of 35 tools skip canonicalisation) and the wrapper already exists; declaration makes the omission impossible rather than discouraged.
Seam ownership: `ToolContext` (server owns; family modules import downward); dispatch wrapper as the one scope/markup gate (`server/markup_guard`); shutdown sequence tuple (main.py owns; `config/schema.py` mirror derives).
Keys: fk-a9f21093207e, fk-d085b903f01c, fk-db357b64fcae, fk-22ab8d1f3a42, fk-1cadfcc36cfa, fk-1509bcd45e2a, fk-45cf67bff540, fk-330a60eef271, fk-9214f842ca92, fk-009fbbf855e7, fk-b3d95cdaa0d3, fk-05a87fd1777d.

## R15 — S16 Orchestrator config shape, defaults and routing (2026-10-10)
Ruling: option 1, pydantic field defaults are the one home of every default; `orchestrator/defaults.yaml` keeps only layered (per-project or environment) values and shrinks to overrides; a parity test asserts no yaml leaf restates a field default (the 23 disagreeing pairs are resolved in the stream's PRD, each with the surviving value named). Rest of S16 as proposed: per-subsystem submodels replacing the 195 prefix-encoded flat fields, with reloadability declared on the field (`RELOADABLE_FIELDS` derived); one role-keyed `RoleSettings` table with `RoleName` and `Backend` as shared StrEnums validated at load and reload (6 knob-major submodels go); the role→route resolver and allowlist moved to shared so fused-memory's 9 model fields and 3 provider vocabularies declare a role and receive a validated route (judge's else→OpenAI arm and the unused adapters go); the generic hot-reload engine hoisted to shared (fused-memory's `reload.py` copy goes); service-path knobs out of `ReconciliationConfig`; consumers trust the validated schema (4 getattr re-validations go); the two double-duty knobs get their own fields; `DEFAULT_ALLOWED_MODELS`/`DEFAULT_LADDER` move to config so routing imports downward (the TYPE_CHECKING cycle goes).
Options considered: (1) field defaults only; (2) defaults.yaml only with required fields.
Consequence that decided it: pyright and the dashboard can read a field default and cannot read a yaml; option 2 makes 146 bare test constructions file-dependent.
Invariant: PROMOTE (synthesis candidate 5) — "a configuration default has exactly one home: no defaults.yaml leaf restates a pydantic Field default" — walk defaults.yaml leaves against `OrchestratorConfig.model_fields` (submodels followed) and assert the restating set is empty; the checkable INV-9 form for config. Owned by S16.
Seam ownership: shared routing resolver + allowlist (shared, S16 owns; orchestrator and fused-memory import); shared hot-reload engine (S16); `RoleName`/`Backend` enums (shared, S16 owns; config.py, invoke.py, invocation_outcome, cli_invoke import — S20 consumes).
Keys: fk-dae2a050b5d3, fk-5805ae6f97f1, fk-2a67692ff456, fk-dac0a2a67201, fk-ba569525f6e0, fk-34d8da6424f4, fk-709d4d84ee37, fk-c520d5374ba5, fk-a3dfde6f0664, fk-efae65b1f9af, fk-da71e5bb1679.

## R16 — S17 Scheduler seams and tick state (2026-10-10)
Ruling: approved as proposed (no open choice): a behaviour-level test builder over `acquire_next`/`release`/`record_requeue`/`set_override` asserting on `get_state_snapshot` and events, so the 1,483 private test references retire file by file; one per-task `TaskRecord` replacing the 18 task-keyed dicts; `TickContext` with required fields and phases invoked by reference; `_deps_satisfied` takes a `DepGates` value (no optional caches selecting its meaning); the stranded-blocked phase consults `TaskGroundTruth.derive_truth` per R6 instead of performing the revert the recovery table forbids; delivered-check evaluation delegates to `delivered_checks.run` as the one evaluator with an event on ERRORED (the silent-forever hold goes); shared `dependency_ids`/`train_id` helpers at the 9 readers; `DeterministicRunner.run` split into predicate/deploy/gate steps over `parse_metadata`; `plan_tools.create_server` takes a `MarkupChannels` bundle; the `_SchedulerLike` shadow protocol and hand-maintained fakes replaced by the typed snapshot seam.
Seam ownership: `get_state_snapshot` as the public test seam (typed via S14's `SchedulerStateSnapshot`); `delivered_checks.run` as the one evaluator (orchestrator owns; landing_evidence and scheduler call).
Keys: fk-a571802ee4ea, fk-4b35b1adea0d, fk-cc01eb7b143f, fk-3e1e7c5129b5, fk-8af92edd6f38, fk-6c103745c784, fk-b2cd27267851, fk-09bf639213ca, fk-a2bc16e5812c, fk-4b3eb71db017, fk-5a5d08e775de, fk-ebce5e23048a, fk-6e7e9bb942d6.

## R17 — S19 Package layering and private seams (2026-10-10)
Ruling: option 1, `session_registry.py` is split in place first (its h6 partition: records and pid pointers, decisions, leases, window helpers, CLI), then the stdlib-only parts move to a shared session package so cockpit depends on shared only; `spawn-claude.sh`'s run-by-path keeps working across both moves. Rest of S19 as proposed: a declared fused_memory subpackage DAG (backends < models/utils < recon_state < services < middleware < reconciliation < server) enforced by a test over the snapshot's typed edge list; the 16 write-path recon modules move to `fused_memory.recon_state`; grouped_read's vocabulary into `memory_metadata` (one `CHILD_KINDS`) and its assembly into services; landlock/bwrap builders and backend resolution into a stdlib-only shared sandbox module imported at module level by both packages (the type-ignored fm→orchestrator imports go); `shared.config_dir.pid_alive` becomes the single public home and the 5 other copies become imports; `_wmctrl_list` and the window-id canonicalisers deduplicated; the 14 cross-module underscore imports (dashboard 22 sites, invoke's 4 cli_invoke privates) become public names or go.
Options considered: (1) split first then move; (2) move whole then split.
Consequence that decided it: option 2 hands shared an h6/h14 alarm that every tier imports; option 1 moves only what belongs.
Invariant: PROMOTE (synthesis candidate 3) — "the fused_memory subpackages form a DAG in the declared order, counting module-level, function-local and TYPE_CHECKING imports" — Tarjan SCC over the snapshot's typed edges restricted to second-level packages, no SCC of size > 1, order a committed list the test reads; the checkable heuristic-9/13 form. Owned by S19.
Seam ownership: `shared.config_dir.pid_alive` (S19); `fused_memory.recon_state` (S19 creates; S12/S13 populate); shared sandbox module (S19; orchestrator invoke and fused-memory sandbox_guard import); shared session package (S19; cockpit and scripts import).
Keys: fk-4fb8ffbb0141, fk-420bc8f9a425, fk-64e06774ee0e, fk-d91317f88f21, fk-cfcf00669c46, fk-e8c835c3f412, fk-57599c7fd907, fk-ea4463b19d86, fk-e22732500172, fk-ad5323d27142, fk-99f44f47a6cc, fk-3ddc73250c64.

## R18 — S20 CLI invocation and usage gate (2026-10-10)
Ruling: option 1, the `UsageGate` slot owns cap detection: `slot.observe(result) -> Decision`, and `invoke_with_cap_retry` becomes a thin state machine over the decisions, decomposed into attempt → classify → decide(policy) → settle with one `cost_store` write per attempt (discarded attempts stop spending invisibly). Rest of S20 as proposed: `shared.invocation_outcome.classify_invocation` (strict-confirm as a parameter) as the one classifier consumed by the loop, `UsageGate.detect_cap_hit` and `_run_probe` (the zero-cost heuristic net and the second classify go); `CliSubtype` StrEnum beside `AgentResult` marking wire vs minted values (5 `error_max_budget_usd` spellings go); `AccountState` transitions only via `_transition` (4 phase setters go; the `cap_hit` event has one writer); a transcript leaf (public locator, content_blocks) replacing 3 nesting-rule copies; `cli_invoke` split per the seat's partition after `CLI_INPUT_REQUIRED_MARKERS` and `classify_agent_failure` move down to break the cycle (5 function-local imports go); the probe uses `resolve_claude_binary` (task 6520 absorbs); scratch root threaded as a parameter; the invocation cost floor gets a home and an enforcement point.
Options considered: (1) slot owns detection; (2) loop owns with `gate.transition(account, phase)`.
Consequence that decided it: "two classifiers remain" is the finding (fk-8805d20c0c4c); option 1 is the only one that leaves one.
Seam ownership: `classify_invocation` (shared, S20 owns; usage_gate and cli_invoke consume); shared transcript leaf (S20; transcript_archive and workflow import); `CliSubtype` (S20; dry_run_unblock, routing, steward, path_scope_adjudicator import); `shared.capability_manifest` public surface (S20 owns; S25 files its `sidecar_for` request as a dependency on S20, per the synthesis's seam assignment).
Keys: fk-1778eb482209, fk-e59c30a25d27, fk-734bc1fc18bf, fk-240df0521f83, fk-dcc5dce60bb6, fk-b8cba955b358, fk-2a5d97aeb64d, fk-8805d20c0c4c, fk-005378a1b462, fk-49b55c1270ae, fk-377f5a42e8be, fk-25c96d016239, fk-f7f1747302dc, fk-db060c262575, fk-ae6cd7600d0e, fk-aacd5e9381f6, fk-3ae32b390a81, fk-fede65204150, fk-2f4b8655fb3c.

## R19 — S22 Dashboard readers and aggregation (2026-10-10)
Ruling: option 1, fused-memory's MCP status tools are extended with the recon / write-journal / ticket panel fields and publish a read model; the dashboard stops naming fused-memory's tables and indexes (13 SQL sites over 8 fused-memory-owned tables and 8 DDL fixtures go). Rest of S22 as proposed: one typed label→root resolver (one|unknown|ambiguous) for scheduler, api/tasks, task_prose and config; one `FANOUT_FAILURE_EXCEPTIONS` classification at every fan-out and one `TTLCache` owner (mcp_fanout split into transport and cache; the four hand-rolled escalation-URL probes collapse); performance's per-DB step returns additive counts with one merge and one partial-tally rule; orchestrator vocabularies (MergeOutcome kind, event types, task uid) imported not re-spelled; one retention SQL home; `app.py` proxies to routers with typed bodies and injected probe state; a fused-memory outage on the curator panel reported as such, never as an empty queue; the 73%/63% drifted prose moves to task records.
Options considered: (1) extend the producer's MCP tools; (2) a shared schema module both import.
Consequence that decided it: option 2 leaves two packages owning one schema behind a parity test; option 1 is the briefing's own "degrade visibly, never synthetic" decision applied at the producer boundary, and a fused-memory migration can no longer break the dashboard silently.
Seam ownership: fused-memory read model for recon/journal/ticket panels (fused-memory S15 owns the tools; S22 consumes); label→root resolver (active_tasks, S22); `shared.project_layout` for DB paths (S14).
Keys: fk-a9e45ca0eb10, fk-389c36ad02d1, fk-70e1ee9218a7, fk-1c1d247caaa7, fk-b5760d0dfe85, fk-94fc76e7a123, fk-8dfa5e5a75aa, fk-c927adb040e9, fk-1b8c06cd8b2f, fk-ce66d4da9b0f, fk-ad49b80c5554, fk-af4ba730a48a, fk-4afbeb684a22, fk-166cacf84a18, fk-d21f64525e52, fk-9979f1b771de, fk-c51423b7b7e5, fk-d0a7ebc4a4f1, fk-6aee6c49fa2e, fk-f3a5a34831e9, fk-1bdd95e804cb, fk-9cf2a161c9c0.

## R20 — S24 Fleet watchdog and redeploy coordination (2026-10-10)
Ruling: option 1, one fleet lease file keyed by tier, in one format shared by bash and the watchdog (the four-tier drift test collapses to one; a third tier is a key). Rest of S24 as proposed: a frozen per-tier supervision spec (unit source, start property, watched paths, clock path, min interval, redeploy unit/script, lease path, streak/cap, storm escape) consumed by one liveness pass and one staleness pass (the 8 sibling pairs, 1,019 lines, and dashboard-watchdog's 5 copies go); the liveness set derived from installed `orchestrator-*.socket` units (the hand-maintained `WATCHED` list goes); a host-reader object injected into each pass (231 private patches retire); fleet paths and lease age in a stdlib-only shared leaf (service_restart, watchdog, df_pytest_isolation import; bash reads via `python3 -c`); `restart-all-orchestrators.sh` acquires the lease under `flock`; watched paths derived from orchestrator config (shared/src included); reclaim-orphaned-worktrees loops over known roots; the 2,015 prose lines to OPERATIONS.md / the PRD.
Options considered: (1) one lease keyed by tier; (2) per-tier files on the fleet template.
Consequence that decided it: option 2 adds the fourth copy the seat warned of and makes every tier change a four-file edit.
Seam ownership: shared fleet-coordination leaf (S24 owns; orchestrator service_restart and scripts import); `scripts/_systemd.py` host reader (S24; both watchdogs import).
Keys: fk-906957d2ec84, fk-c9390bf37295, fk-ed19c0c4d4df, fk-3830e2a6de79, fk-68efae8a36fc, fk-1cc71283157c, fk-4c21aaf01b82, fk-47a7eb8c020d, fk-63633270af4e.

## R21 — The uncontested block: S11, S14, S18, S21, S23, S25 (2026-10-10)
Ruling: all six approved as proposed, each one `/prd` session in an agent team. Ordering: S14 (shared leaves) is wave 1 because S12, S13, S17, S19 and S25 consume its leaves; S25 depends on S20's `shared.capability_manifest` leaf; the others are independent.
- S11 Curator pipeline and candidate types: approved as proposed. Keys: fk-f61accce2f37, fk-1e0486bf9ab9, fk-4f895cd425c2, fk-caf8bcd5b7ed, fk-756aef0b35b7, fk-85234d9d39c4, fk-b259ba3af605.
- S14 Shared leaves for cross-package facts: approved as proposed. Keys: fk-06de98fa328b, fk-3644c6841b3e, fk-7a85d17a7c7e, fk-e367d6a83223, fk-1b25c0b5d3a2, fk-43a75be2cd47, fk-f6ab6bfad0be, fk-37f2baab3aec, fk-b84e0aa4e545, fk-f5e16692e170, fk-7aff55dd1465, fk-75fa713ba3d5, fk-f01bf7bc23f9, fk-954f6787b07c, fk-9cdaf62ed9af, fk-ccd26ad88810.
- S18 Evals measurement integrity: approved as proposed. Keys: fk-0d4c1014e41f, fk-3505aba0481d, fk-28ca5c64de06, fk-43584c88970d, fk-b80c8e52e526, fk-b6d389a4fcf2, fk-2e8daab1913f, fk-294ee8dfa70f, fk-df50bc042300, fk-b2c28b4f8f17, fk-14926fe151d4, fk-4352b1ee49f5.
- S21 Legibility pipeline: approved as proposed. Keys: fk-3a23d0a51e69, fk-f1314d3b25eb, fk-74142ca0f882, fk-e3cc7c87d73e, fk-15f40965b6ba, fk-da87d818b1b8, fk-988f412eced2, fk-ee9a4bae4e83, fk-ef642b412580, fk-3ea2b2df348a, fk-12f0d08a6ccd, fk-c9ab2773a5d9, fk-0531ecaefba5, fk-0af74b3620ec, fk-234e0a63dfd7, fk-a0af0f46c1dc.
- S23 Sampler collection groups and schema: approved as proposed. Keys: fk-e86a10b9e217, fk-37f494b26355, fk-ab2c587ff5a9, fk-d207683eef5a, fk-be1621618dee, fk-fcefc2576467, fk-895f269d7bb5.
- S25 Scripts instruments: shared locators and parity checkers: approved as proposed. Keys: fk-3042baa5983e, fk-638eb09e52b1, fk-b16470a6e0d2, fk-7fee08736560, fk-9a99057011f3, fk-cb9f382ab2bd, fk-679f3dc5ca48, fk-f70004144a3d, fk-30cd73379093, fk-915ca17415b7, fk-9bb2996a620a, fk-195f86e31fea, fk-f4af080fcd25.

## R22 — Invariant candidates (2026-10-10)
All seven synthesis candidates PROMOTED, each in checkable form with a fixture, each through the CONTRIBUTING.md §6 four-site lockstep edit, each owned by one stream; the program doc carries a pointer per promoted INV id and never a second list (INV-9):
1. every escalation record was written through the one filing entry point — S1 (R8).
2. no module outside escalation/ compares a status/action/category to a string literal — S2 (R9).
3. the fused_memory subpackages form a DAG in the declared order, counting module-level, function-local and TYPE_CHECKING imports — S19 (R17).
4. parse_metadata's registered slice set is a property of shared.task_metadata alone (fresh-interpreter parity) — S10 (R3).
5. a configuration default has exactly one home (no defaults.yaml leaf restates a Field default) — S16 (R15).
6. a merge gate that cannot measure returns GateMeasurement.unmeasurable and emits merge_gate_unmeasured; no gate returns its pass value from an rc != 0 or except branch — S7 (R11).
7. the sampler's emitted metric vocabulary, admission sets, gate-arm table and dashboard known metrics are projections of one MetricGroup table — S23 (R21).
Plus the lane timing-budget ordering validator from R1 (S6), which is a configuration invariant rather than a code-shape one: eight promotions in all. INV ids are minted by the lockstep edit in each owning stream's PRD, not here.

## R23 — Weakened findings (2026-10-10)
Ruling: all 14 weakened findings (6 ranked in a cell's top three: fk-06de98fa328b, fk-a9e45ca0eb10, fk-1c1d247caaa7, fk-37f494b26355, fk-fcefc2576467, fk-240df0521f83; 8 weakened mechanical left unfiled: fk-f12106b3adaf, fk-04c4612589ea, fk-5564cc32de5c, fk-6b0dec845d9e, fk-c0932f0b7df1, fk-0165547376d8, fk-b49ccd852736, fk-c9935b02f118) are routed to their owning streams; each `/prd` session re-verifies its weakened keys under G3/G6 before filing or dropping; dispositions stay `open` until then. No acceptance filed.
Options considered: route all; accept the three thinnest as standing; accept all 14.
Consequence that decided it: none has a reason worth recording as an acceptance, and the PRD gates re-verify weakened claims anyway.

## R24 — Launch plan (2026-10-10)
Ruling: wave 1 now = the six agent-mode streams (S11, S14, S18, S21, S23, S25) as one `/team` run, one `/prd` author+decompose seat per stream (S25 waits on S20's leaf for `sidecar_for` and files that part as a dependency). The 20 spawn-mode streams are launched in later sittings from this program doc, high-risk first (S4, S5, S6, S8, S9, S10, S26), then the rest. The trigger chain (interim form) is filed when the first stream files and committed only when the FILED table is complete.

## R25 — S3 merge tools wait for the native lane-query port; no interim adapter (2026-11-10)
Ruling: option A (sitting key S3-1). The two merge-tool moves — S3 ε1 9188 (read/cancel/relane tools) and S3 ε2 9193 (`merge_request`) — wait for S6 μ 9201, which builds `MergeLaneQueries` natively over the lifecycle registry. No `SnapshotLaneQueries` adapter exists anywhere. S3 α 9152 shrinks to the Protocol, the two state records, the query-unavailable error and the dedupe rename; the three snapshot-dict parsers in `merge_lane/intake.py` stay until ε2, which converts the dedupe gate onto the native port, deletes them and adds the `lane_unanswerable` answer field. S6 μ depends on S3 α only and must never depend on an S3 merge-tool move (that edge closes a cycle). Tasks 5409 (tip_sha on every workflow_verify green) and 4651 (periodic landed reconciler) have their edges reversed: each lands first in today's files and the move carries its code (S3 ε2 → 5409; S3 ε1 → 4651). The history index θ 9177, the gate tables ι 9181 and the header-constants task κ 9173 are ordered ahead of the ladder split ζ 9200 so they are not held. S6 μ's priority comes from the scheduler's priority inheritance through the dependency graph; it is not hand-raised.
Options considered: (A) wait for the native implementation (ruled); (B, recommended) ship a typed interim adapter, `SnapshotLaneQueries` over `snapshot()`, deleted later by S6 μ; (C) build the native version inside S3.
Consequence that decided it: Leo chose A without a stated reason beyond the choice; the measured effect in `digests/discussion/S3.md` §S3-1 (two live fixes held behind ~20 tasks, and a cycle) was accepted together with the mitigations above, which are part of the ruling.
Supersedes: none (R10 named the port, not an interim form of it).
ruled_by: human
Carried by: `plans/review-all-s3-escalation-server-and-queue-boundary-prd.md` α 9152, ε1 9188, ε2 9193; `plans/review-all-s6-merge-lane-lifecycle-outcome-types-and-provenance-prd.md` μ 9201; existing tasks 5409, 4651.
Keys: fk-c8c3406fd7f4, fk-0de72a91f89a, fk-a8d3003e9d83, fk-e903cc072b15, fk-f8e9e0da3786, fk-d9e6c03dc259; fk-bc64099a1cd0 is `filed:5409`.

## R26 — S3 history index covers the archive tier only; the fused-memory writer sentence corrected (2026-11-10)
Ruling: option A (sitting key S3-4): index the archive tier only; the pending tier stays the root glob. The plan's false sentence is corrected everywhere: each project's queue directory has two long-running writers — that project's orchestrator (which hosts the escalation MCP server) and the single system-wide fused-memory service — restarting on separate clocks. The integration signal names a fused-memory restart through `scripts/restart-fused-memory.sh` or its staleness backstop, not the fleet sweep.
Options considered: (A, recommended) index the archive only, pending stays a scan (ruled); (B) the index serves every read, pending too; (C) a SQLite index database; (D) narrow the scan by file name.
Consequence that decided it: Leo asked whether fused-memory is not one system-wide process; it is (one singleton per host, the counted modules all inside it). The corrected picture makes B safer during rollout but not safe, and it still buys almost nothing (`digests/discussion/S3.md` §S3-4).
Supersedes: none.
ruled_by: human
Carried by: S3 θ 9177 and the S3 integration gate μ 9210.
Keys: fk-e5d81cc1f1d4, fk-ce2dc99e0305.

## R27 — One internal merge-lane state enum and one total mapping to the public `MergeState` (2026-11-10)
Ruling: the plan's recommendation for sitting key S3-5 is superseded by Leo. Inside the orchestrator there is one internal enum, `ItemLifecycleState`, and one total, test-checked mapping to the public `MergeState`; an unrecognised state cannot occur, and the internal enum is not made the public vocabulary. The port record carries `state: MergeState` (no raw state crosses the port; `raw_state` is dropped); the enum-to-enum table lives in `merge_lane/queries.py` and is built by S6 μ 9201; `MergeCancelVerdict.state` is typed (S3 ε1 9188); the pinned pass-through test is replaced by an exhaustiveness test. Task 6827 is cancelled as covered.
Options considered: (A, recommended) a typed state that may be empty with the raw string kept beside it; (B) treat an unknown state as 'cannot answer'; (C) map an unknown state to a generic 'unknown'; superseded by Leo with one internal enum and one total mapping.
Consequence that decided it: the worker already holds a closed enum and renders strings only because the escalation server could read nothing but a dictionary; once the merge tools move into the orchestrator (R10, R25) both sides are in one package, so one enum and one total mapping are possible (`digests/discussion/S3.md` §S3-5).
Supersedes: none.
ruled_by: human
Carried by: S6 μ 9201; S3 ε1 9188.
Keys: fk-d9e6c03dc259, fk-a8d3003e9d83, fk-e903cc072b15, fk-f8e9e0da3786.

## R28 — `VerifyHost` with `host` required on every verify function (2026-11-10)
Ruling: option A (sitting key S4-1), a `VerifyHost` value, with Leo's change: `host` is REQUIRED on every verify function (entry points and public probes), with no default anywhere inside the verify modules. The process host is resolved once, by `VerifyHost.process()`, at the long-lived boundary objects the harness builds (task workflow, merge-lane worker, review checkpoint, eval runner) and at the two remote verify CLI commands. Two guards: no `host` default in any verify signature, and production code outside the composition roots never calls `VerifyHost.fresh()`. The ContextVar stays only as the process slot and the per-test reset.
Options considered: (A, recommended) a process-wide `VerifyHost` value (drafted with `host` optional); (B) a runner keyword, caches stay global; (C) the runner as a sixth context field; A with `host` required (ruled).
Consequence that decided it: every production caller wants the same process host, so the real risk is verify code that forgets to pass its host on, which in tests silently drops the scripted runner and runs real commands; a required parameter turns that omission into a type error (`digests/discussion/S4.md` §S4-1).
Supersedes: none.
ruled_by: human
Carried by: `plans/review-all-s4-verify-run-context-and-runner-contract-prd.md` ι1 9134 (lands C5), ι3 9141 (moves it), κ 9142 (extends it); S8 β 9218 keeps the one `VerifyHost.process()` call in the composition root.
Keys: fk-126a75a57ca5 (κ; ι1 and ι3 carry no key).

## R29 — A bare worker crash is re-run once at both gates, then is a real failure (2026-11-10)
Ruling: new option ruled by Leo (sitting key S4-5). At BOTH gates (task verify and merge gate) a bare worker crash is re-run once; a second crash is a real failure whose report names the crash and the test running at the time, and it goes to the fix loop — the debugger at task verify, the steward at the merge gate — never to a human hold. The bare-crash category is not infra-transient. The cross-check fleet halt behaves as today.
Options considered: (A, recommended) route merge-gate crashes to the infra hold; (B) the merge gate keeps blaming the branch; (C) an isolated rerun decides first; Leo's question whether task verify should also fail on a crash; re-run once then a real failure at both gates (ruled).
Consequence that decided it: today's split has no principled reason (in July 2026 the merge path had no handler for the raise, a reason now gone), and failing on every crash would bring back the debugger churn on overload crashes that the task-path change stopped (`digests/discussion/S4.md` §S4-5).
Supersedes: R4 — a clarification, not a reversal: R4 retired the raise and made the crash data; this rules where that datum goes.
ruled_by: human
Carried by: S4 α 9128; S9 κ 9241 keeps the rule on the workflow side.
Keys: fk-5037fafc755c, fk-09ad1a19f587.

## R30 — `orchestrator verdict-parity` command (2026-11-10)
Ruling: option A (sitting key S4-8; the recommendation was to defer). Expose `verify_runner.py::run_verdict_parity` as the CLI command `orchestrator verdict-parity`, defaulting to the single divergent SHA; rewrite both escalation action texts to name it and `clear_verify_host_quarantine`. One new leaf.
Options considered: (A) expose it as a CLI command (ruled); (B) delete it and change the procedure; (C, recommended) leave it for a later run.
Consequence that decided it: Leo chose to give the operator the tool the escalation text points at; the discussion confirmed it buildable in a CLI process (`digests/discussion/S4.md` §S4-8).
Supersedes: none.
ruled_by: human
Carried by: S4 ρ 9166.
Keys: fk-e906c4e914c2.

## R31 — Task 5040 finishes before S6's lifecycle work (2026-11-10)
Ruling: option B (sitting key S6-2): task 5040 (the verify-dispatch extraction) finishes first and nothing parks it. S6 δ 9171, ζ 9182 and κ 9192 depend on 5040 and build on its result (δ because 5040's part 2 moves the arm δ rewrites); 5040 waits on no S6 or S7 task, which voids the lead's earlier S7/OQ-3 edge set onto 5040. Every other clause of R1 stands.
Options considered: (A) hold 5040 and rebase it later onto S6's lifecycle work; (B) let it finish first (the lead's changed recommendation for the sitting; ruled); (C) hold it for the first phase only.
Consequence that decided it: 5040 was dispatched 2026-11-03/04 and holds a half-built branch (5 commits, 2,315 lines added), which is the plan's own stated fallback trigger; the lead changed the recommendation to B for the sitting and Leo accepted it.
Supersedes: R1 — its order only ("then `ItemLifecycle` as the operative state … then the collaborator decomposition"), for task 5040 alone. Recorded by the lead.
ruled_by: human
Carried by: S6 δ 9171, ζ 9182, κ 9192; existing task 5040. Measured consequence: 92 of the 162 wave-2 tasks now sit behind 5040 (measured 2026-11-11; rule and breakdown in Program status, open items).
Keys: fk-7f2bce2d482c, fk-75fc1b96098d, fk-313f404f3cc6, fk-a2984bf98f85, fk-769c3bb58ce1, fk-201d70aaf409, fk-e7deb8bcfa95.

## R32 — Lifecycle refusals: escalate now, stop the lane after a clean 14 days (2026-11-10)
Ruling: option B staged into stop-the-line (sitting key S6-5). On landing, a refused lifecycle transition is a typed refusal that escalates and the lane continues. After the operative registry has run 14 days with zero refusals on dark-factory and reify, a refusal places a per-lane holder of kind `lifecycle_refused` owned by the refusal's escalation and leaves the request in place. One restart-only config value, `merge_lane.lifecycle_refusal_policy: escalate | hold_lane` (default `escalate`), is flipped per project by a dated deterministic gate task. Storm guard: one holder per lane per refused edge. A `lifecycle_refused` holder is rehydrated from its pending escalation across a restart, so a redeploy never silently restarts a stopped lane (lead, as authored). Decision 4 of `plans/merge-queue-modularization-invariants-prd.md` gets a written amendment naming this invariant.
Options considered: (A) fail the request on refusal; (B, recommended) typed refusal, escalate and continue; (C) fail early, continue late; (D) refuse and leave it in place; Leo's question whether a broken invariant should halt and be fixed at once (Toyota stop-the-line); B staged into stop-the-line at lane scope (ruled).
Consequence that decided it: refusals are rare (25 in total, all 2026-07-11 to 2026-08-07, none in the last 90 days), about two thirds pointed at real lane defects, and escalate-and-continue let one real defect sit unfixed for 90 days (task 4151); staging guards against a storm caused by a bug in the registry's own table (`digests/discussion/S6.md` §S6-5).
Supersedes: decision 4 of `plans/merge-queue-modularization-invariants-prd.md` (escalate-and-continue becomes stage 1 only); extends R1 by giving R1's operative `ItemLifecycle` a refusal policy R1 did not state (R1's text otherwise stands).
ruled_by: human
Carried by: S6 ζ 9182 (typed refusal, default branch), θ 9176 (holder kind), φ 9186 (policy value, hold arm, storm guard, rehydration, `scripts/lifecycle_refusal_census.py`, the decision-4 amendment). NOT filed yet: ξ (the 14-day quiet-period gate: deterministic, milestone delayed 1209600 s, predicate `scripts/lifecycle_refusal_census.py` over dark-factory and reify `runs.db`, `--window-days 14`) and υ (arm `hold_lane` in `dark-factory-orchestrator.yaml`), which the lead files once 9186 is on main (R58); after ξ is done the lead commits reify's one-line yaml flip; the other seven projects stay `escalate`.
Keys: fk-313f404f3cc6, fk-a2984bf98f85, fk-83625b943aa5, fk-ba531799522b.

## R33 — Deep chain is kept; only the probe and auto-chain are deleted (2026-11-10)
Ruling: option A (sitting key S7-1): keep deep chain; delete the variable-depth probe (~275 lines) and auto-chain (~194 lines). The lane keeps adjacent placement plus deep chain. Deep chain is owned by the deep merge-ahead and merge-lane policy plans; merge-lane policy tasks 8527 and 8528 join S7's same-file neighbours; task 6242 is amended (it listed finishing auto-chain as a possible fix). S6 re-measures any count that assumed ~1,450 deleted lines.
Options considered: (A, recommended) keep deep chain, delete the probe and auto-chain (ruled); (B) delete deep chain too, as R11 ruled; (C) keep deep chain behind a new placement port.
Consequence that decided it: R11's premise, "every project config has them off", still holds for the probe and auto-chain but has been false for deep chain since 2026-10-20 (reify) and 2026-10-27 (dark-factory), and Leo has since ruled twice more to build on deep chain (`digests/discussion/S7.md` §S7-1).
Supersedes: R11 — the deletion of the deep-chain placement mechanism only.
ruled_by: human
Carried by: `plans/review-all-s7-merge-lane-gates-placement-mechanisms-and-target-adapters-prd.md` β 9133; existing tasks 8527, 8528, 6242.
Keys: fk-7d1580bb3f03.

## R34 — Reify retry narrowing is deleted end to end (2026-11-10)
Ruling: option B (sitting key S7-3; the recommendation was to move it behind an adapter). Delete reify retry narrowing end to end: the producer, the narrowed-retry budget, the flag, the `merge_request` tool parameter (`retry_failed_only`), the event marker fields, the narrowed-retry arm of the warm shadow, tests, fixtures and docs. No `target_verify.py` module is created. Task 6576 is cancelled as moot; task 6575 (measure-then-decide on narrowing, temporarily parked on 5040) is cancelled with `x_acceptance_reason` naming the deletion task and this ruling. The deletion lands before S3's `merge_request` move (S3 ε2 9193 → S7 δ 9127); S4 drops its `build_retry_env` consumer row.
Options considered: (A, recommended) move it behind `build_retry_env` in a target-verify adapter; (B) delete the feature end to end (ruled).
Consequence that decided it: "preserving the code is not worth the effort and complexity" (Leo); measured, 0 of 5,408 merge verifies since 2026-07-20 were narrowed and no caller sets the flag (`digests/discussion/S7.md` §S7-3).
Supersedes: R11 — the remedy clause "reify retry narrowing behind `build_retry_env` in a target-verify adapter module with a contract pointer" (the ~700 reify-specific lines still leave the worker, by deletion); and Leo's 2026-10-07 ruling on task 6285 (dormant becomes removed).
ruled_by: human
Carried by: S7 δ 9127 (first in S7's chain, lead sequencing R54); existing tasks 6576, 6575 (cancels).
Keys: fk-dd411ada7c29.

## R35 — Durable per-task steward spend and a $25 default cap (2026-11-10)
Ruling: option A (sitting key S8-1): steward lifetime spend is durable per task; task 3566 is cancelled. PLUS Leo: the default cap rises to $25 (`orchestrator/src/orchestrator/config.py::OrchestratorConfig.steward_lifetime_budget` Field default and `orchestrator/src/orchestrator/defaults.yaml`, both 12.0 today), and this repo's own yaml line (20.0) is deleted with its comment, ALL IN THE SAME COMMIT as durable spend; no other project sets it. Gate task 5822's budget item is amended to a read-only spend-distribution report.
Options considered: (A, recommended) durable per-task spend, cancel the rename task 3566 (ruled, with Leo's addition); (B) rename the key, leave spend per-instance; (C) durable spend held until the steward-mode gate rules.
Consequence that decided it: Leo asked to raise the default and remove lower caps; the measurement found only this repo sets one and no other dollar cap binds a task's total steward spend below $25 (`digests/discussion/S8.md` §S8-1).
Supersedes: none.
ruled_by: human
Carried by: `plans/review-all-s8-harness-composition-root-and-slot-registry-prd.md` θ 9229; existing tasks 3566 (cancel), 5822 (amend).
Keys: fk-a1085af3a290.

## R36 — One event writer, and the orchestrator reads events only through `EventStore` (2026-11-10)
Ruling: option A (sitting key S8-3): batched full-durability writes, reads drain first. PLUS Leo: encapsulate the orchestrator process now — `digest.py` and `strict_event_reads.py` move onto typed `EventStore` read methods, the offline orchestrator tools (`storm_window_bound.py` included) are allow-listed explicitly, and a guard test fails on a raw `FROM events` anywhere in `orchestrator/src` except `event_store.py`. The "flush at named sites" alternative is dropped as subsumed. The cross-package `runs.db` read model (dashboard, fused-memory, client feed, quality series, scripts) is an S22 follow-up (wave 3; seam table), not an S8 task.
Options considered: (A, recommended) batched full-durability writes, reads drain first (ruled, with Leo's addition); (B) batched writes with an explicit flush at named sites; (C) relax to `synchronous=NORMAL` under WAL; (D) a persistent connection only.
Consequence that decided it: raw-SQL readers are a missing abstraction (Leo); only two raw readers run inside the orchestrator process at runtime, and no in-process design can give the out-of-process readers read-your-writes, so the ban is scoped to `orchestrator/src` and the rest goes to S22 (`digests/discussion/S8.md` §S8-3).
Supersedes: none.
ruled_by: human
Carried by: S8 ι 9230, ι2 9231; S22 (follow-up, unfiled until S22 is authored).
Keys: fk-f1a7f57fa58e.

## R37 — `convert_to_blocked_enforce` is deleted, in a safe order (2026-11-10)
Ruling: option B (sitting key S8-7; the recommendation was A), in Leo's confirmed order: (1) a tiny independent first task flips the field's code default to True; (2) after that is deployed fleet-wide the LEAD removes the key from the eight other project yamls (autopilot-video, know-live, pump-web-ui, reify, scoping-intake, solar-challenge, solar-challenge-platform, stream-touch), best effort, one yaml-only `git commit --only` per repo, recording each sha or skip reason in S8 δ's description — a lead action with no task dependency; (3) the rule-table task deletes the field and this repo's yaml line. A project missed gets one self-closing unknown-key L2.
Options considered: (A, recommended) delete the log-mode rewrite and keep the field as deprecated and ignored, a later task deleting it; (B) delete the field outright (ruled, in the order above); (C) keep it as a classifier policy; (D) flip the default now, delete the field later; Leo's addition: a milestone to edit every other project's yaml before the deletion deploys, best effort.
Consequence that decided it: the code default is False (log mode), so a yaml line removed before the default flips puts that project in log mode until the deletion lands; flipping first makes the yaml edits safe at any time (`digests/discussion/S8.md` §S8-7).
Supersedes: none.
ruled_by: human
Carried by: S8 η 9212 (default flip), δ 9219 (deletion); the lead's eight yaml edits after 9212 deploys.
Keys: none of its own (η carries none; δ's keys are R6's).

## R38 — The harness fails closed when a store cannot be built (2026-11-10)
Ruling: option A, fail closed (sitting key S8-8; the recommendation was B, stand-in stores). `Harness.__init__` builds `EscalationQueue` first, then the event, run and cost stores with no degrade; it retries only SQLite locked/busy (bounded, about 3 attempts over about 30 s); on failure it files one born-at-L2 critical escalation on a fixed sentinel naming the store, path and exception, logs ERROR and raises. The stand-in store classes are dropped. The composition-root task depends on the watchdog crash-loop alarm (task 8270).
Options considered: (A) fail closed (ruled); (B, recommended) do-nothing stand-in stores, loudly; (C) keep the optional stores.
Consequence that decided it: "fail loud and hard, do not degrade" (Leo); the 8270 edge and the L2 before the raise keep a bad `runs.db` from becoming a quiet systemd restart loop (`digests/discussion/S8.md` §S8-8).
Supersedes: none.
ruled_by: human
Carried by: S8 β 9218 (depends on 8270).
Keys: fk-d8f08696dd66, fk-fc3c4cba6a0f, fk-5d0a66a2d32e.

## R39 — The state machine records a failed row write and keeps going (2026-11-10)
Ruling: option B (sitting key S9-5, as recommended), with one added contract sentence: done and cancelled writes stay OUTSIDE the state machine and keep `Scheduler.mark_done`'s raise-on-failure. Leo also ruled a follow-up outside S9: a periodic (about 2-minute) pass over the existing landed-merge journal with a one-hour give-up to an L1, after task 4646, listed with S8's sweeps — a program-level follow-up the lead files or routes, not a task of S9.
Options considered: (A) raise on a failed write; (B, recommended) record the failure and keep going (ruled); Leo's open side question: also queue and retry failed writes.
Consequence that decided it: the machine never writes done (the store's done paths own it); over 30 days no dark-factory done write was lost to a transient failure, and the failures were refusals a retry cannot fix (`digests/discussion/S9.md` §S9-5).
Supersedes: none.
ruled_by: human
Carried by: `plans/review-all-s9-workflow-phase-spine-and-recovery-ledger-prd.md` ν 9284; the landed-journal pass is unfiled (Program status, open items).
Keys: fk-ee708be4436f, fk-f37f7e253cd5.

## R40 — The L0 settle move is measured 14 days after it lands (2026-11-10)
Ruling: option A (sitting key S9-7, as recommended) PLUS Leo's check: a small early script task, `scripts/check_l0_settle_outcomes.py` (read-only, with a test), on which the steward-outcome task depends; and a deterministic delayed-milestone check task 14 days after the steward-outcome task lands, measuring stranded or wrongly dismissed blocking L0s with the thresholds in `digests/discussion/S9.md` §S9-7 (baseline: 3.6 harness-stranded blocking L0s per 100 filed; 1 wrong dismissal in 30 days). Large numbers mean a joint redesign with the harness. The check task is filed by the lead once the script is on main.
Options considered: (A, recommended) the steward decides, the workflow believes it, one settle (ruled, with Leo's check); (B) full ownership with a terminal hand-off.
Consequence that decided it: the store refuses a deterministic task whose script does not yet exist, so the script lands first and the check is filed after it (`digests/discussion/S9.md` §S9-7).
Supersedes: none.
ruled_by: human
Carried by: S9 π 9164 (script, high), ε 9190 (steward outcome; depends on π); the 14-day check is unfiled (R58).
Keys: fk-bab6058b23c8, fk-e39550272df9.

## R41 — The status-transition rule is enforced in the backend only (2026-11-10)
Ruling: new option ruled by Leo (sitting key S10-3). One verdict function in `shared.task_transitions`, one call site in the backend's status writers inside the write transaction, `actor` required, the reopen reason passed. The interceptor's terminal-exit gate and legality gate are deleted; the interceptor renders the backend's refusal through the one error renderer and keeps the tokens.
Options considered: (A, recommended) one verdict with today's policy in both layers; (B) the backend refuses every illegal change; (C) the backend refuses terminal exits only; Leo's question whether the backend should enforce at all; backend only (ruled, a new option).
Consequence that decided it: every routine status write already passes the interceptor and only three paths write straight to the backend, none illegally; the backend floor protects against future direct writers, so one enforcement point in the backend is enough (`digests/discussion/S10.md` §S10-3).
Supersedes: R3 — its clause "the status FSM enforced in the backend as well as the interceptor" becomes "in the backend only". The rest of R3 stands.
ruled_by: human
Carried by: `plans/review-all-s10-task-write-contract-metadata-modes-success-signal-and-registry-prd.md` κ1 9248; ν 9282 (the `StatusTransitionGate` partition adapts).
Keys: fk-d80c1c44bca1, fk-1eb04ceac9e8, fk-e3e99447e11e.

## R42 — `metadata.files_to_modify` is destroyed (2026-11-10)
Ruling: new, ruled by Leo (sitting key S10-5). Fused-memory refuses an incoming metadata payload carrying `files_to_modify` (submit and update; incoming keys only) with an error naming `files`; the shared declared-files reader reads `files` only; every other reader of the key is deleted; a test fails if any module under `*/src` reads it. The lead backfills the three rows that differ (dark-factory 2507, reify 4344 and 4865 as of 2026-11-10) after the refusal deploys. The precedence question is withdrawn. The details-text prefix stays as planned (stop writing, age out). Renaming the curator's internal field is an optional low follow-up outside S10.
Options considered: (A, recommended) keep the store's precedence rule between the two keys; (B) adopt the interceptor's skip-empty rule; search-and-destroy the key (Leo, ruled).
Consequence that decided it: the key is nearly extinct — 4 rows across all 11 stores, all done, the newest from July, no code writing it — so destroying it is mostly deleting readers and the precedence question goes moot (`digests/discussion/S10.md` §S10-5).
Supersedes: none.
ruled_by: human
Carried by: S10 δ 9204, ε 9209; the lead's three-row backfill after 9204 deploys.
Keys: fk-fa3e3d765f50, fk-fa355c2c0498.

## R43 — One rule for a human gate task (2026-11-10)
Ruling: new, ruled by Leo (sitting key S10-6). A task is a human gate when it is deterministic AND `always_escalates` is true (stored shape; for a submission, after the submit-time coercion). Refinements are named properties (pure vs act-then-ask; a curator gate is also `operational_mode == 'gate'`). The blocked-gate audit and the curator sweep keep their own honestly named selectors; the four other consumers move onto the one rule. Live signal: a one-week count of extra recon duplicate-gate refusals.
Options considered: (A, recommended) one home, the named rules unchanged; (B) every consumer uses the interceptor's rule; one uniform rule, to be chosen (Leo) — ruled as above.
Consequence that decided it: the interceptor's rule over-reports (14 tasks it calls gates dispatch to an agent), while the orchestrator hands a task to a human exactly when it is deterministic and always-escalating (`digests/discussion/S10.md` §S10-6).
Supersedes: none.
ruled_by: human
Carried by: S10 λ 9254 (`shared.gate_declaration`).
Keys: fk-06ab2d3e9148.

## R44 — Candidate-key collisions refused with their own token and counted after 30 days (2026-11-10)
Ruling: option A (sitting key S10-7, as recommended) PLUS Leo's check: a collision in the curator combine is an audited refusal with the distinct token `candidate_key_collision` (with the survivor id and both file sets); a census script, `scripts/combine_collision_census.py`, must be on main before the check task is filed; a deterministic delayed-milestone check 30 days after the combine task counts occurrences across all projects, and "frequent" (≥ 5 collisions or ≥ 1% of combine writes) means address it more elegantly.
Options considered: (A, recommended) refuse, record the survivor, create (ruled, with Leo's check); (B) retarget the combine into the survivor; (C) refuse and drop the candidate.
Consequence that decided it: Leo wants the frequency measured before a redesign; a deterministic task's script must exist on main at filing (`digests/discussion/S10.md` §S10-7).
Supersedes: none.
ruled_by: human
Carried by: S10 ζ 9214 (combine), ρ 9237 (census script); the 30-day check (S10 σ) is unfiled (R58).
Keys: fk-de3df95984a5.

## R45 — The memory service's mutators split out past the identity third (2026-11-10)
Ruling: Leo ruled three extra tasks (sitting key S26-2) after the backends-private task and before the integration test: (1) move the write-metadata policy and store-codec helpers out of `memory_service.py` (`memory_metadata_policy.py`, `store_records.py`); (2) extract `GraphMutations` over a small shared journaled-operations module, `journaled_ops.py` (`journaled_op`, `journaled_backend_call`, event sink; NOT inside `write_journal.py`); (3) extract `MemoryMutations` (after 1). `MemoryService` keeps thin public delegates. A follow-up records the read-path split (`memory_reads.py`) and a prose pass. Same-file edges or notes go on the live tasks that name moved mutators.
Options considered: (A, recommended) stop at the identity split; (B) also move the metadata policy; (C) also move the maintenance mutators; a deeper clean decomposition (Leo) — ruled as the three tasks above.
Consequence that decided it: measured, the graph mutators are already almost independent of the write path, and the plan's reason for not splitting them stops being true once the plan lands (`digests/discussion/S26.md` §S26-2).
Supersedes: none — it carries R7's partition past the identity third R7 named and does not change R7's choice of the reconciler; it moves `journaled_op`'s home to `journaled_ops.py`.
ruled_by: human
Carried by: `plans/review-all-s26-memory-service-write-path-and-graph-client-prd.md` ρ 9243, τ 9245, υ 9247, φ 9252 (follow-up, dispatched after the integration gate π 9250).
Keys: fk-da5b47d0493a, fk-343c4a42be6b, fk-50cc281018fb.

## R46 — The cross-process identity exclusion is designed in a human session (2026-11-10)
Ruling: option D (sitting key S26-3, as recommended) as a HUMAN DESIGN SESSION: the follow-up is a deterministic, always-escalating gate task asking for an interactive `/spawn` `/team --fable` session (investigate, discuss designs, answer every open question before the PRD) with a committed brief file, `plans/fm-cross-process-identity-exclusion-session-brief.md`; it is not an architect task. Lead sequencing: the gate depends only on S26's public-identity-lock task and absorbs task 6293's design question; task 6730 must not depend on the gate.
Options considered: (A) accept the race as standing; (B) build a cross-process lease now; (C) run cross-graph moves inside the server; (D, recommended) file a follow-up that owns it (ruled, as a human design session).
Consequence that decided it: Leo wants the design discussed before a PRD exists; the gate costs no agent dispatch (`digests/discussion/S26.md` §S26-3).
Supersedes: none.
ruled_by: human (sequencing: lead)
Carried by: S26 σ 9198 (high, so gate 6698 waits on it; depends on λ 9194); existing tasks 6293, 6730.
Keys: fk-9804ecc4567d.

## R47 — Task 4867 is hand-carried and S10's first task waits on it (2026-11-10)
Ruling: task 4867 (a write-nothing `update_task` fails; unknown MCP arguments are loud) is hand-carried now by Leo's ruling; S10's first task depends on 4867 as originally drafted. The lead's inversion of that edge, ruled earlier the same day because 4867 was lock-starved, is void.
Options considered: invert the edge for 4867 alone (the lead's ruling); hand-carry 4867 and keep the drafted edge (Leo).
Consequence that decided it: hand-carrying removes the lock starvation that motivated the inversion.
Supersedes: the lead's S10/OQ-7 ruling of 2026-11-10 (`digests/wave2-lead-rulings.md`), not an R entry.
ruled_by: human
Carried by: S10 α 9180; existing task 4867 (deferred under its carry).
Keys: fk-ff5234ac1013, fk-116481623cfb.

## R48 — Wave-2 design questions accepted as recommended (2026-11-10)
Ruling: every question below was accepted as the plan recommended ("Otherwise, your recommendation LGTM"); the PRD text authored under the recommendation stands and is the record of what was decided. Sitting keys, with the plan's question id:
- S3: S3-2 (OQ-2, all three emitters write `merge_deduplicated`), S3-3 (OQ-5).
- S4: S4-2 (OQ-2), S4-3 (OQ-3), S4-4 (OQ-4), S4-6 (OQ-7), S4-7 (OQ-8, option b).
- S5: S5-1 (OQ-3, A′: task 4752 folded into λ 9267), S5-2 (OQ-5, B′, including the low-priority re-entrant lane-lease follow-up σ 9277), S5-3 (OQ-9).
- S6: S6-1 (OQ-1; tasks 4581, 7396 cancelled into α 9140), S6-3 (OQ-3), S6-4 (OQ-4; task 5046 cancelled), S6-6 (OQ-10; acceptance record 9211, cancelled with `x_acceptance_reason`), S6-7 (OQ-11; live effect accepted: dark-factory no-progress budget 5400 s → 7800 s, hung-verify hold 4.5 h → 6.5 h, reify 9.5 h), S6-8 (OQ-12, option b).
- S7: S7-2 (OQ-2), S7-4 (OQ-CQ1), S7-5 (OQ-CQ3).
- S8: S8-2 (OQ-2; 4651 depends on ε2a 9223), S8-4 (OQ-5), S8-5 (OQ-6; confirms the lead's reading of R6's "both recovery sweeps": `CrashRecovery` and `StrandedReconciler`), S8-6 (OQ-7).
- S9: S9-1 (OQ-1), S9-2 (Q-C1), S9-3 (OQ-2), S9-4 (OQ-3), S9-6 (OQ-4), S9-8 (OQ-6), S9-9 (Q-C4).
- S10: S10-1 (OQ-3), S10-2 (OQ-4), S10-4 (OQ-8; task 3260 amended).
- S26: S26-1 (OQ-2), S26-4 (OQ-8), S26-5 (OQ-9), S26-6 (OQ-10; task 5612 cancelled into ξ 9215), S26-7 (OQ-11).
Options considered: as each plan's `## Rulings (sitting of 2026-11-10)` section states.
Consequence that decided it: Leo found each recommendation sound; none reverses an R entry.
Supersedes: none.
ruled_by: human

## R49 — R5's git-runner wording clarified (2026-11-10)
Ruling: the `WorktreeMissing` adaptation is the leaf `orchestrator/git_run.py::run_in_worktree`; `git_ops._run` is deleted (S5/OQ-6, option A; ruled by the lead 2026-11-02 and shown to Leo in the sitting's uncontested block 2026-11-10).
Options considered: keep the adaptation in `git_ops._run`, as R5's letter says; a leaf module (ruled).
Consequence that decided it: R5's intent is one `WorktreeMissing` adaptation over `shared.git_async.run_git`; the lead recorded the leaf as satisfying that intent (S5 PRD §Rulings OQ-6, D2, C1).
Supersedes: R5 — wording only: "git_ops keeps only the `WorktreeMissing` adaptation" is read as its intent (one adaptation site), not its letter (the site is `git_run.py`).
ruled_by: lead
Carried by: `plans/review-all-s5-gitops-ports-query-contract-and-the-git-runner-prd.md` α 9255, β 9258 (C1).
Keys: fk-d73aeaebb792, fk-3a4cc6298c02.

## R50 — R11's resource-audit fingerprint clarified (2026-11-10)
Ruling: resource-audit violation records are fingerprinted on (kind, subject, grace_secs, detail) (S7/OQ-CQ6; ruled by the lead, shown to Leo 2026-11-10).
Options considered: R11's literal (kind, subject); the four-field form (ruled).
Consequence that decided it: the four-field form keeps task 3203's pinned grace and permit-growth re-log.
Supersedes: R11 — clarification of "fingerprinted on (kind, subject)".
ruled_by: lead
Carried by: S7 ε 9149.
Keys: fk-5d4406afeade.

## R51 — Invariant landing order INV-15 → INV-16 → INV-17 (2026-11-10)
Ruling: S6 α 9140 (`lane-budgets-exceed-verify-timeouts`) lands first as INV-15; then S10 θ 9242 (`metadata-slices-one-table`) as INV-16; then S7 ζ 9154 (`merge-gate-unmeasured-typed`) as INV-17. Edges on the three invariant files: S10 θ depends on S6 α, S7 ζ on S10 θ. Every check anchors on the slug, never the number. S6's separate invariant docs leaf was folded into α.
Options considered: the lead's earlier order (S10 θ → S7 ζ → S6 α); this order.
Consequence that decided it: the budget validator fixes a live misconfiguration (a 5400 s budget under a 7200 s timeout) and R1 made it a first leaf; the earlier order held it behind S10's deep chain and all of S7 (the S10 critic's delta pass N1).
Supersedes: the lead's invariant numbering of 2026-11-02 (`digests/wave2-lead-rulings.md` "Invariant numbering (S10/OQ-5)"); R22 set no order.
ruled_by: lead
Carried by: S6 α 9140, S10 θ 9242, S7 ζ 9154.
Keys: fk-4b76c9695674, fk-f528087f44c7, fk-88e3c228b649.

## R52 — Tasks 4833, 5409 and 4651 land first in today's files (2026-11-10)
Ruling: three live tasks land before the wave-2 moves that would otherwise carry them: 5409 (S3 ε2 9193 depends on it) and 4651 (S3 ε1 9188 depends on it), both from Leo's S3-1 mitigations (R25); and 4833 (delete `TerminalOutcomeRetention`), by the lead: S3 ε1 9188 and S6 κ 9192 depend on 4833, 4833 waits on no S3 or S6 task, and S6 α is not held behind it. `fk-66746efddd4d` is `filed:4833`; `fk-bc64099a1cd0` is `filed:5409`. Task 4651 additionally depends on S8 ε2a 9223 (R48, S8-2), and its temporary parking edge on 5040 was removed at S8's decompose. Task 4567 is a prerequisite of S3 θ 9177 only.
Options considered: each task waits for the move and is re-scoped onto the moved code; each lands first and the move carries its code (ruled).
Consequence that decided it: holding three live fixes behind the merge-tool moves (which wait on S6's native-port chain and on 5040) delays them by ~20 tasks for no design gain.
Supersedes: none (the plans' earlier edge directions only).
ruled_by: lead (5409 and 4651: human, R25)
Carried by: existing tasks 4833, 5409, 4651, 4567; S3 ε1 9188, ε2 9193, θ 9177; S6 κ 9192; S8 ε2a 9223.
Keys: fk-66746efddd4d, fk-bc64099a1cd0.

## R53 — Task 6236 is kept as an extraction and lands before S10's interceptor partition (2026-11-11)
Ruling: task 6236 (done-provenance extraction) is KEPT, amended to "extraction only" (the git-runner seam half belongs to S5 μ 9276), and S10 ν 9282 depends on 6236: 6236 lands first, and `StatusTransitionGate` imports `done_provenance.py` downward. 6236 depends on task 5274; if 5274 stalls, the fallback is to cancel 6236 into ν.
Options considered: cancel 6236 into S10 ν; keep it after ν (the S5 seat); keep it before ν (the S10 seat; ruled).
Consequence that decided it: ν's rule that it imports nothing back from `task_interceptor.py` would otherwise force ν to move the done-provenance cluster itself (`critique/S10.align.md` §Not applied).
Supersedes: none.
ruled_by: lead
Carried by: existing task 6236 (its text refreshed at filing); S10 ν 9282; S5 μ 9276.

## R54 — Wave-2 decompose procedure and the one per-file order (2026-11-11)
Ruling: all nine wave-2 plans were filed IN PARALLEL in planning mode with intra-plan and existing-task edges only, no seat committing; the lead then wired every cross-plan edge from `digests/wave2-order.json` (in the program state home; built from the nine `critique/<S>.align.md` chains, the live store's edges and the rulings; verified acyclic: 165 tasks, 702 edges, 90 per-file edges on the 73 non-test files shared across plans), re-checked the live store for cycles, and only then were batches committed. The per-file order is one global topological order — free tasks first, then tasks behind 8528, then tasks behind 5040; within a level S6's first task, then priority, then seam order S5 S4 S10 S7 S3 S6 S9 S8 S26 — and on each non-test file shared across plans consecutive tasks in that order are chained; test files are left to the scheduler's file locks. Rulings applied within it: S6 α 9140 lands first on every file it shares with another plan and takes no edge on S5 α 9255, S5 λ0 9256, S7 δ 9127, S7 β 9133, S3 or task 4833 (S3 α 9152 → S6 α); S7 δ heads S7's chain (δ → β → α1 → α2 → γ → ε → ζ); S5 λ0 is second in S5's chain; S5 ν 9274 keeps its chain position, so S9 δ 9184 onward waits on 5040 through S5 (accepted); the S8 ↔ S3 cycle cut (S8 β 9218 depends only on S3 β 9157 and δ 9167; S3 ε1 depends on S8 ε2b 9224; S8 ε3 9225 and ω 9236 depend on S3 ζ 9200); S8 ε2b depends on S9 θ 9213, one way.
Options considered: decompose seat by seat in seam order, each committing its batch (the protocol's default).
Consequence that decided it: `add_dependency` does not reject cycles, and a batch committed before its cross-plan edges exist can dispatch early.
Supersedes: none (deliberation.md §Launching step 3's sequential form, for this wave).
ruled_by: lead
Carried by: all 162 wave-2 tasks.

## R55 — New worker.py tasks park behind 8528, except S6's first task (2026-11-11)
Ruling: every new wave-2 task declaring `merge_lane/worker.py` depends on task 8528, extending Leo's 2026-11-10 chain protection (which parked every pending worker.py task behind 8528) to the program's new tasks so none takes the lock at a handoff. EXCEPTION: S6 α 9140 (the timing-budget validator, INV-15, a small fix of a live misconfiguration) is NOT parked, because it is ruled first on every shared file and parking it would put about 50 otherwise free tasks behind the chain. One edge reverses the exception; it is flagged to Leo (Program status). Measured side effect: S7 δ 9127, and S3 ε2 9193 through it, now wait on 8528 → 8527 → 7720, unlike S7's PRD sentence "δ does not wait on 7720".
Options considered: park every new worker.py task; park none; park all but 9140 (ruled).
Consequence that decided it: the chain protection is Leo's; the single exception is the one task whose delay costs a live misconfiguration and ~50 tasks.
Supersedes: none.
ruled_by: lead
Carried by: the parking edges on 8528 in S3, S4, S5, S6 and S7; S6 α 9140.

## R56 — S26's graph-access task waits on task 7743 (2026-11-11)
Ruling: S26-D1 = (a): S26 θ 9179 depends on task 7743, which lands `backends/paged_read.py` first; `graph_access` imports it. Edge wired by the lead with the cross-plan wiring.
Options considered: (a) θ depends on 7743 (ruled); (b) cancel 7743 into θ; (c) leave them unordered under file locks.
Consequence that decided it: both tasks re-home `graphiti_client._paged_ro_query`; 7743's graphiti_core-free `paged_read.py` keeps `falkor_read_only` from loading the FalkorDB driver, which a home inside `graph_access.py` would undo, so 7743 owns the home and `graph_access` imports it.
Supersedes: none.
ruled_by: lead
Carried by: S26 θ 9179; existing task 7743.

## R57 — S8's Scheduler-construction codemod keeps its declared files (2026-11-11)
Ruling: S8-d1: task 9217 (S8 β0, the Scheduler-construction codemod) keeps its 73 declared files as the plan rules; it is hand-carried if lock-starved for more than 24 h. Mentioned to Leo with the 5040 question (same lock-starvation class).
Options considered: keep the 73 files with a hand-carry trigger (ruled); declare only the 13 non-codemod files and let the architect widen.
Consequence that decided it: the codemod touches every one of the files, so an undeclared edit would be a scope violation; the lock-starvation risk (as with 5040 and 4867) is met by the hand-carry trigger.
Supersedes: none.
ruled_by: lead
Carried by: S8 β0 9217.

## R58 — Dated check and policy-flip tasks are filed only after their prerequisite lands (2026-11-11)
Ruling: four ruled tasks are deliberately NOT filed at decompose; the lead files each under `planning_mode` with its `prd_path`/`prd_task_label` once its trigger holds: S6 ξ and υ (R32) after S6 φ 9186 lands; S10 σ, the 30-day combine-collision check (R44), after S10 ρ 9237 (`scripts/combine_collision_census.py`) is on main; the S9 14-day L0 check (R40) after S9 π 9164 (`scripts/check_l0_settle_outcomes.py`) is on main.
Options considered: file them now with an edge on the script task (refused by the store); file later (ruled).
Consequence that decided it: `submit_task` validates a deterministic task's `before_done.script`, so a check filed before its script is on main fails at submit.
Supersedes: none.
ruled_by: lead
Carried by: S6 φ 9186, S10 ρ 9237, S9 π 9164 (the triggers).

## R59 — The idle-workflow slot leak fix lands before the slot-registry rewrite (2026-11-11)
Ruling: task 5153 (a registered-but-idle workflow is invisible to all three reapers and holds its slot; high, no prerequisites) lands first, in today's code. The S8 slot-registry task 9216 depends on 5153 and folds the container 5153 adds into `SlotRegistry`. The edge 9216 depends on 5153 was wired on 2026-11-11.
Options considered: A the fix lands first and the registry task absorbs it (ruled); B the fix waits for the registry task, as the S8 draft suggested.
Consequence that decided it: on 2026-11-11 task 9216 had 207 unfinished transitive prerequisites, including task 5040, so option B would have left a live slot leak open behind a long refactoring chain.
Supersedes: the S8 PRD's recommended edge "5153 depends on the slot-registry task" (a recommendation, never a ruling; never kept in the store).
ruled_by: human (Leo, in the first lead's thread, 2026-11-11: "agreed" on "the fix lands first"; record `~/.claude/review-all/review-all-dark_factory-20261008/digests/leo-answers-via-predecessor-2026-11-11.md`)
Carried by: task 5153; S8 α 9216 (`plans/review-all-s8-harness-composition-root-and-slot-registry-prd.md`).

## R60 — Task 5040 is hand-carried once the merge-lane policy chain's last task is done (2026-11-11)
Ruling: task 5040 (the verify-dispatch extraction; 90 declared files, a half-built branch) is carried by a dedicated session through the merge queue as soon as task 8528 is done. It is not carried ahead of the chain. A scheduled check launches the carry and stands down if one already exists.
Options considered: A hand-carry after 8528 (ruled); B leave it to the factory and carry only after 24 hours of lock starvation; C hand-carry now, ahead of the policy chain.
Consequence that decided it: 92 of the 162 wave-2 tasks wait on 5040 (measured 2026-11-11, R31 made it a prerequisite), and a 90-file task starved for file locks from 2026-11-04; carrying it ahead of the chain would take the merge worker file from the chain protected on 2026-11-10.
Supersedes: none (it completes R31 and follows the precedent of R47).
ruled_by: human (Leo, in the first lead's thread, 2026-11-11: "do", confirmed "do this"; same record file)
Carried by: task 5040; brief `~/.claude/spawn-briefs/hand-carry-5040.md`.


## Shared conventions (every session)

- Read this document first, then the stream's keys in `plans/review-all-dark_factory-20261008.md`
  (`## Findings by area` and `## Candidate streams`), then `docs/code-quality.md` and
  `docs/quality-findings-contract.md` §6–§8.
- **Freshness clause.** Every anchor was verified at `6551b75aba6b1885a2aea4bfc46ac82446fba4d4` and main
  has moved. Before building a task on an anchor: `git diff 6551b75aba..main -- <path>` and re-read the
  symbol. A weakened key (R23) is re-verified under G3/G6 before any task is filed on it; drop it with a
  one-line note in the PRD if the cost is gone.
- **Filing stamps.** Every task carries `metadata.x_finding_key` (a list of the keys it closes),
  `metadata.x_finding_run = "review-all-dark_factory-20261008"`, `metadata.source = "review-all"`,
  file-level `metadata.files`, and `priority` = the finding's severity (contract §4). File under
  `planning_mode=True` → `add_dependency` → one `commit_planning` per batch; verify the batch with
  `get_task`, never `search_tasks`. Run the contract §8 dedup protocol per key before filing (step 1: the
  forensic `json_each(metadata, '$.x_finding_key')` query; step 2: `search_tasks` ≥ 0.6 confirmed by
  `get_task`): a Phase-5 mechanical task for the same key is a dependency or is cancelled with
  `x_acceptance_reason`, never duplicated.
- **Acceptance.** A cost the PRD decides not to remove is a `cancelled` task carrying the key and
  `metadata.x_acceptance_reason`; never `deferred`.
- **Trigger chain.** The contract §11 human gate for this run is filed in the interim form
  (`scripts/check_run_completion.py` is absent at as_of) by the lead when the first stream files; each
  later session adds its `high`/`critical` tasks as dependencies of the gate id recorded in the FILED
  table (`add_dependency(id=<gate>, depends_on=<task>)`); `commit_planning` on the gate happens only when
  this table is complete.
- **Commits.** PRDs land in `plans/<slug>-prd.md` with their capability manifests; commit with
  `git commit --only <paths>` (the main checkout is machine-operated and the merge lane is live).
- **Seams.** Touching another stream's seam is a dependency on that stream's leaf, never a parallel
  edit (S25↔S20 on `shared.capability_manifest` is the precedent, resolved in the synthesis).

## Triage briefing

For the L2 watcher while this program lands: a `MetadataWriteFailed` (S10) surfacing as a block is the
designed behaviour, not a regression; `merge_gate_unmeasured` events (S7) replace silent passes and
should be read as the gate reporting, not failing; the lane timing-budget validator (S6) refuses a yaml
that raises a verify timeout above the budget at config load, which is the intended loud failure.

## Program status (2026-11-11)

Read by the next `/review-all`'s Phase 0 (contract §9). Wave-1 figures are from the 2026-11-01 store sweep
(`find_tasks_by_metadata(key="x_finding_run", value="review-all-dark_factory-20261008")`: 118 tasks then,
confirmed with `get_task`) and were not re-swept; wave-2 figures are from a 2026-11-11 store read
(`get_statuses` over the 162 filed ids, `get_task` on gate 6698). The live tooling under
`skills/review-all/scripts/program/` is still not usable against this server until task 7178 lands (7178
`pending` on 2026-11-11).

**Wave 1 — done.** Deliberation (R1–R24, 2026-10-10) and wave 1 of R24: the six agent-mode streams S11,
S14, S18, S21, S23 and S25 were authored, critiqued and decomposed on 2026-10-10/11 (PRDs and stamped
sidecars in `plans/review-all-s{11,14,18,21,23,25}-*-prd.md`, commits `4cfbb33757`, `ae3047ec3a`,
`81731dc052`). Those six streams filed 60 tasks (59 leaves plus 6709, an acceptance record); as of
2026-11-01: 31 done, 26 pending, 2 held `deferred` (6764, 6770), 1 cancelled-as-accepted (6709). One foreign
leaf, 6861 (`plans/offline-lane-red-path-prd.md` δ2, done), also carries the run id because it absorbed
mechanical 6639's key. S23 is complete. Phase 5 filed 55 mechanical tasks 6587–6646 (as of 2026-11-01: 16
done, 36 pending, 1 in-progress (6615), 2 cancelled (6639, 6644)). Two skill defects found during the run
carry the run id without a finding key (6785 done, 6786 pending as of 2026-11-01).

**Wave 2 — authored, ruled, filed.** Wave 2 is S3 S4 S5 S6 S7 S8 S9 S10 S26 (S3 and S7 were taken into it
from the Streams table's wave 3). The nine plans were authored and critiqued to convergence on 2026-11-02,
ruled by Leo at the sitting of 2026-11-10 (R25–R48; the lead's rulings are R49–R58), re-folded under the
rulings, fold-verified, then decomposed in parallel and filed under one cross-plan edge wiring (R54): 162
tasks, ids 9127–9289 except 9211, plus the cancelled acceptance record 9211 (S6). All 162 were `pending` on
2026-11-11; none is held. The 45 high leaves are dependencies of gate 6698. The PRDs, capability manifests
and stamped sidecars are on main (`53c17d997f` PRDs, `eedd7cde9a` sidecar stamps, `e22018b057` S9 sidecar fix,
`11bb013b5c` three delivered checks widened ahead of wave-2 moves, tasks 7053 and 9203). Wave 2 owns three of R22's
invariants, landing in the order INV-15 (S6 9140), INV-16 (S10 9242), INV-17 (S7 9154) (§Design invariants
promoted). Tasks deliberately not filed yet, each filed by the lead when its trigger holds (R58): S6's two
dated policy-flip gates (ξ, υ) after 9186 lands; S10's 30-day combine-collision check after 9237's script is
on main; S9's 14-day L0 check after 9164's script is on main. Program-level follow-ups not yet filed or
routed: the periodic landed-journal pass (R39, after task 4646, beside S8's sweeps); the shared `runs.db`
events read model, which goes to S22 (R36; seam table); one mechanical task for S4's residual
fk-591206484ea5; the reify-side removal of the `speculation_probe` stanza before S7 β 9133 deploys; the
optional low curator-field rename (R42). The clerk pass of 2026-11-11 (program state home, `clerk/`)
applied the decompose returns' actions on pre-existing tasks: 191 applied (cancels, among them 3566, 5612,
6575, 6576, 6827, 5046, 4752, 6914 and 7115; amends, among them 5822, 6242, 3260, 6236 and 6719; the
finding-key stamps on 5409, 4833 and 5697), 5 skipped (notes on done, in-progress or branch-carrying tasks
6861, 2507, 3010, 6497 and 5040), 2 left to the lead (edge 5153 → 9216, never kept: ruled the other way by R59; the 4686 stamp, since
applied by the lead) and 2 rejected (a placeholder note for the unfiled S9 check; a note on gate 6698).
Lead actions still open: the eight yaml edits of R37, the three-row backfill of R42, and the unfiled
follow-ups and dated checks above.

**Wave 3 — not authored.** S1, S2, S12, S13, S15, S16, S17, S19, S20, S22 and S24 (S22 is agent-mode wave 1
in the Streams table but was left out of R21/R24's wave-1 six) have no PRD and no task; their 164 finding
keys (the Streams table's per-stream counts summed) stand in `plans/review-all-dark_factory-20261008.json` as
the run's `open` findings or as `re-observed` `prior_rows`. Four of R22's promoted invariants are owned by
wave-3 streams and do not exist yet: the filing entry point (S1), the escalation literal ban (S2), the
fused_memory subpackage DAG (S19) and the config default single home (S16). R23's eight weakened
mechanical keys (fk-f12106b3adaf, fk-04c4612589ea, fk-5564cc32de5c, fk-6b0dec845d9e, fk-c0932f0b7df1,
fk-0165547376d8, fk-b49ccd852736, fk-c9935b02f118) belong to no stream in the Streams table, so "routed to
their owning streams" routed them nowhere; no wave-2 plan took them, and they remain unfiled and open for a
closing screen. The two S25 held leaves still wait on wave 3: ι 6764 on S20 (C3) and ξ 6770 on S2 (C5), both
`deferred` on 2026-11-11; task 7724 depends on held 6764; the S25 sidecar carries `task_id: null` for ι and
ξ, as designed for a held leaf. S2 and S20 release them at their decompose.

**Triage briefing.** `## Triage briefing` describes behaviour of S6, S7 and S10; their tasks are filed but
none has landed (S6 α 9140, S7 α1 9136/α2 9139, S10 β2 9191 deliver it).

**Open items for Leo.**
1. **Hand-carry task 5040** (the verify-dispatch extraction: 90 declared files, a half-built branch
   `task/5040` with 5 commits ahead of main, lock-starved since 2026-11-04). Under R31 it finishes first,
   and 92 of the 162 wave-2 tasks sit behind it. Rule (read-only query of the store's `dependencies`
   table, 2026-11-11): a wave-2 task counts when a chain of dependency edges leads from it to 5040 and no
   task on that chain is `done` or `cancelled`. By stream: all 20 of S4; 21 of S8's 22 (all but η 9212);
   18 of S5's 23 (17 of its 21 chain tasks, plus σ 9277); 17 of S9's 21 (δ 9184 onward); 8 of S6's 13
   (δ, ζ, θ, φ, κ, λ, μ, ι); S3 ε1, ε2, ζ, λ, μ; S7 γ, ε, ζ; none of S10 or S26. 5040 itself depends on
   8528 (`pending`) under Leo's 2026-11-10 chain protection, besides 5305 (`done`). RULED 2026-11-11 (R60): it is hand-carried
   once 8528 is done.
2. **Task 5153 versus the slot-registry task 9216**: RULED 2026-11-11 (R59). Task 5153 lands first; 9216
   depends on it (edge wired 2026-11-11). The reverse edge the S8 draft suggested is withdrawn.
3. **The 9140 parking exception** (R55): S6 α 9140 is the one new worker.py task not parked behind 8528.
   Behaviour-relevant; one edge reverses it.

**Gate.** Human gate 6698 (interim form, delayed milestone 259200 s, `pending` since its 2026-11-01 release
on Leo's instruction) depends on 51 tasks: wave 1's 6707, 6721, 6736, 6740, 6747, 6750 and wave 2's 45 high
leaves. Of wave 1's, 6707, 6736, 6740 and 6747 were `done` and 6721 (waits on 6720, `pending`) and 6750 (waits on 6749, `in-progress`) still `pending` on
2026-11-11; every wave-2 dependency is `pending`, S26 σ 9198 (the human design-session gate) among them. When
the gate fires, the next run re-derives streams from the open findings per contract §9; wave 3's high leaves
join it as each wave-3 stream files.

## FILED — program status

Appended by each `/prd` session as it files; completed by the 2026-11-01 cleanup sitting with one row per
stream; the nine wave-2 rows (S3 S4 S5 S6 S7 S8 S9 S10 S26) replaced their "not authored" rows on
2026-11-11 from the decompose returns. Columns: stream · PRD path · task anchors (ids) · gate dependencies
added · notes. Wave-1 leaf status is as of the 2026-11-01 store sweep; wave-2 leaf status as of the
2026-11-11 store read (all 162 pending). The trigger-chain gate id, the superseded chain (none: first run) and the
`review_due` escalation resolved (none: hand launch) are recorded here by the lead.

| Stream | PRD | Tasks | Gate deps added | Notes |
|---|---|---|---|---|
| (trigger chain) | — | human gate **6698**: `metadata.trigger_chain.skill=review-all`, `run_id=review-all-dark_factory-20261008`, `task_kind=deterministic`, `always_escalates`, milestone delayed 259200 s; deps 6707, 6721, 6736, 6740, 6747, 6750 | — | interim form (`scripts/check_run_completion.py` still absent 2026-11-01); no prior chain to supersede; launched_from none; `commit_planning(task_ids="6698", target_status="pending")` run 2026-11-01 by the cleanup sitting on Leo's instruction — echo `deferred → pending`, `deferral_cleared: null`; the delayed milestone runs from that release. 2026-11-11: the 45 high leaves of the nine wave-2 streams were added (deps now 51 — the six above plus 9130, 9135, 9138, 9140, 9156, 9157, 9159, 9160, 9164, 9167, 9168, 9170, 9175, 9176, 9179, 9180, 9182, 9184, 9187, 9188, 9190, 9191, 9192, 9193, 9194, 9195, 9198, 9200, 9206, 9207, 9210, 9215, 9219, 9223, 9224, 9240, 9241, 9242, 9246, 9249, 9260, 9261, 9267, 9284, 9286); 6721 and 6750 were still pending, so the milestone had not started |
| S11 curator-pipeline-and-candidate-types (agent, wave 1) | plans/review-all-s11-curator-pipeline-prd.md | α 6703 (low) done, β 6704 done, γ 6705 done, δ 6706 done, ε 6707 (high) pending, ζ 6708 pending; ω acceptance 6709 cancelled (fk-85234d9d39c4, `x_acceptance_reason` present) | 6698 → 6707 | 4/6 leaves done; ε's deps (6705, 6706, 4503) are all done, so it is dispatchable; ext deps 4513 (α, landed), 6385 (ζ); 6176 cancelled (→β, reason stamped); 6332 depends on 6708; priority inversion α low → ε high recorded, no override (contract §4) |
| S14 shared-leaves-for-cross-package-facts (agent, wave 1) | plans/review-all-s14-shared-leaves-prd.md | done: α 6710, ζ 6711, λ 6712, ε 6713, β 6714, γ 6715, η 6717, π 6725; pending: δ 6716, θ 6718, ξ 6719 (low), ν 6720, μ 6721 (high), κ 6722, ι 6723, ο 6724, ρ 6726 (integration gate) | 6698 → 6721 | 8/17 done; μ waits on ν 6720; fk-06de98fa328b dropped (cost gone, task 4618) — the stream's only key with no task; 4709 depends on 6718; 6333 cancelled (→6720, reason stamped); ζ 6711 widened to SyncMcpSession per S21's extent; the 10-11 OPEN item is closed: 6713 depends on 6758 (S25 γ) and both are done — the sidecar's ε `scripts-checkout-bootstrap` verdict still reads OPEN (stale, not re-stamped); 6726 allow-list references S21's transport leaf 6747 (done) |
| S18 evals-measurement-integrity (agent, wave 1) | plans/review-all-s18-evals-measurement-prd.md | done: α 6733, β 6734, γ 6736 (high), δ 6737, ε 6738, ζ 6739, η 6741 (low), θ 6743 (low); pending: ι 6746 (low), κ 6748 (low), λ 6752 (low) | 6698 → 6736 | 8/11 done, every remaining leaf low; 6324 cancelled (→6737/6738, reason stamped); λ depends on 6597 (done) and 6710 (done); the S16/S20 same-file edges on orchestrator/cli.py were never wired because S16/S20 were never authored (no harm observed: β/ε/θ/ι/κ landed or wait on nothing from them) |
| S21 legibility-pipeline (agent, wave 1) | plans/review-all-s21-legibility-pipeline-prd.md | done: α 6740 (high), β 6742, γ 6744, δ 6745, ε 6747 (high); pending: ζ 6749, η 6750 (high), θ 6751, ι 6753 | 6698 → 6740, 6747, 6750 | 5/9 done; η waits on ζ 6749; cancelled (→leaf, reasons stamped): 5960→6740, 6357→6742, 6466→6747, 5927→6749, 6009→6750, 4241→6750; 5693 (done) kept as a dependency of 6740; 5785 depends on 6740, 5513 on 6744; 6747 is the transport leaf S14's 6726 allow-list references; census-incremental §4.7 cross-reference to D8/6749 is in `plans/census-incremental-prd.md` (landed) |
| S23 sampler-collection-groups-and-schema (agent, wave 1) | plans/review-all-s23-sampler-metric-groups-prd.md | α 6772 done, β 6773 done (INV `sampler-vocabulary-one-table` minted) | none (no high leaves) | STREAM COMPLETE; 6644 cancelled (→6773, reason stamped); 6643 (pending, mechanical) depends on 6772; β depended on 6758 (done) |
| S25 scripts-instruments-shared-locators-and-parity-checkers (agent, wave 1) | plans/review-all-s25-scripts-instruments-prd.md | done: α 6756, β 6757, γ 6758 (bootstrap), ν 6769; pending: δ 6759, ε 6760, ζ 6761, η 6762, θ 6763 (low), κ 6765 (low), λ 6766, μ 6767 (low); HELD deferred, never released: ι 6764 (`x_seam_wait` S20/C3; sidecar `task_id: null`), ξ 6770 (`x_seam_wait` S2/C5; sidecar `task_id: null`; esc-3169-2 declare_pin migration is its step 1) | none (no high leaves) | 4/14 done; the two holds wait on streams S20 and S2 that were never authored — release needs a ruling (see Program status); 7724 (pending) depends on held 6764; 6713 (S14 ε) and 6723 → 6758 edges present; 6037 amended, 6764 depends on it (and on θ 6763); 6770 depends on ε 6760; 6447 cancelled (→6756) WITHOUT `x_acceptance_reason`; 6381 depends on 6756; 6422 depends on 6769; PRD text lags rulings 3/6/7 in four places (task descriptions + sidecar authoritative) |
| S1 one-escalation-filing-entry-point (spawn, wave 3) | — | — | — | not authored — R24 later sitting never held; 11 keys open; owns R22 invariant 1 (R8) |
| S2 typed-escalation-record-facts (spawn, wave 3; upstream S1) | — | — | — | not authored — R24 later sitting never held; 14 keys open; owns R22 invariant 2 (R9); S25 ξ 6770 is held on its C5 leaf |
| S3 escalation-server-and-queue-boundary (wave 2 by the 2026-11-10 sitting; Streams table: spawn, wave 3; upstream S6, S2) | plans/review-all-s3-escalation-server-and-queue-boundary-prd.md | 12 leaves, ids 9152–9210 (not contiguous; the nine streams were filed in parallel), all pending 2026-11-11: α 9152, β 9157 (high), γ 9162, δ 9167 (high), κ 9173 (low), θ 9177, ι 9181, ε1 9188 (high), ε2 9193 (high), ζ 9200 (high), λ 9205, μ 9210 (high) | 6698 → 9157, 9167, 9188, 9193, 9200, 9210 | none held. ε1 9188 and ε2 9193 wait on S6 μ 9201 by dependency (R25), ζ λ μ behind them; existing-task edges κ → 3176, θ → 4567, ε1 → 4651/4833/8528, ε2 → 5409 and → S7 δ 9127 (R34, R52). Keys: fk-bc64099a1cd0 → `filed:5409`, fk-66746efddd4d → `filed:4833` (both stamped by the clerk pass 2026-11-11); 6827 cancelled as covered (R27). WorkflowSlots declared in δ 9167 (seam table). Integration gate / signal leaf μ 9210. No invariant (R22 none). |
| S4 verify-run-context-and-runner-contract (wave 2; Streams table: spawn, wave 2, high-risk) | plans/review-all-s4-verify-run-context-and-runner-contract-prd.md | 20 leaves, ids 9128–9169 (not contiguous; the nine streams were filed in parallel), all pending 2026-11-11: α 9128, β 9129, γ 9130 (high), δ 9131, ζ 9132, ι1 9134, η 9135 (high), θ 9137, ι2 9138 (high), ι3 9141, κ 9142 (low), λ 9146, μ 9148, μ0 9150, ν 9153, ξ 9156 (high), ε 9160 (high), ο 9163 (low), ρ 9166, π 9169 | 6698 → 9130, 9135, 9138, 9156, 9160 | none held. fk-5fa21a4b85cb is `fixed:31d2464f04` (no leaf); fk-3756ad2f3357's builder half is `fixed:fca89bd55b` (task 7918), its policy half in ζ 9132 (which also absorbs 5615); fk-591206484ea5 residual stays open: the lead files one mechanical task for it (not yet filed). Rulings R28 (ι1 9134), R29 (α 9128), R30 (ρ 9166). Parked on 8528: α β η θ ι1 μ0 ξ ρ (R55). Integration gate / signal leaf π 9169. Every S4 task waits on 5040 through S6 (R31, Program status). No invariant. |
| S5 gitops-ports-query-contract-and-the-git-runner (wave 2; Streams table: spawn, wave 2, high-risk) | plans/review-all-s5-gitops-ports-query-contract-and-the-git-runner-prd.md | 23 leaves, ids 9255–9277 (contiguous), all pending 2026-11-11: α 9255, λ0 9256, β0 9257, β 9258, γ 9259, δ 9260 (high), ι 9261 (high), κ 9262, ε 9263, ζ 9264, η 9265, θ 9266, λ 9267 (high), ξ0 9268, ξ 9269, ο0 9270, ο 9271, ρ0 9272, ρ 9273, ν 9274 (low), π 9275, μ 9276 (low), σ 9277 (low) | 6698 → 9260, 9261, 9267 | none held. Chain α → λ0 → β0 → β → γ → δ → ι → κ → ε → ζ → η → θ → λ → ξ0 → ξ → ο0 → ο → ρ0 → ρ → ν → π, plus μ and the low re-entrant lane-lease follow-up σ 9277 (no key; depends on θ 9266; R48 S5-2). fk-54eb9f7cc0be is `filed:5697` (`x_finding_key`/`x_finding_run` stamped on 5697 by the clerk pass 2026-11-11; its `source` stays `steward-sweepup`); S6's fk-59b3df6f52a3 is disposed in η 9265; 4752 cancelled into λ 9267, 6914 cancelled into ε 9263. R49 (git runner, α/β). Eight leaves parked on 8528. At `commit_planning` three substrate checks were refused as `vacuous_present` (already true on main) and not copied into task metadata: α 9255 `substrate-run-git-exists`, ι 9261 `substrate-lifecycle-exists`, κ 9262 `substrate-acquire-route-enum`; every id is stamped. 17 of its 21 chain tasks wait on 5040 through S6 ζ 9182 (R31). No invariant. |
| S6 merge-lane-lifecycle-outcome-types-and-provenance (wave 2; Streams table: spawn, wave 2, high-risk; upstream S5, S7) | plans/review-all-s6-merge-lane-lifecycle-outcome-types-and-provenance-prd.md | 13 leaves, ids 9140–9206 (not contiguous; the nine streams were filed in parallel), all pending 2026-11-11: α 9140 (high), β1 9144, β2 9155, γ1 9159 (high), γ2 9165, δ 9171, θ 9176 (high), ζ 9182 (high), φ 9186, κ 9192 (high), λ 9197, μ 9201 (low), ι 9206 (high) | 6698 → 9140, 9159, 9176, 9182, 9192, 9206 | none held. Acceptance record 9211 (S6-6): filed and cancelled with `x_acceptance_reason`, keys fk-c55b12f4b66d, fk-e8a03e4ec067 (the coalesce-train arm of fk-ba531799522b named in the reason). fk-d41caf6b6331 is `fixed:9d0dfb0079`; fk-59b3df6f52a3 → S5 η 9265. α 9140 mints INV-15 `lane-budgets-exceed-verify-timeouts` (R51) and is the one new worker.py task NOT parked on 8528 (R55); β1 γ1 δ θ ζ φ κ λ are parked. δ 9171, ζ 9182, κ 9192 → 5040 (R31). 4581, 7396 (into α) and 5046 cancelled. NOT FILED: ξ (14-day quiet-period gate) and υ (dark-factory `hold_lane` flip), filed by the lead after φ 9186 lands (R32, R58). Integration gate / signal leaf ι 9206. |
| S7 merge-lane-gates-placement-mechanisms-and-target-adapters (wave 2 by the 2026-11-10 sitting; Streams table: spawn, wave 3) | plans/review-all-s7-merge-lane-gates-placement-mechanisms-and-target-adapters-prd.md | 7 leaves, ids 9127–9154 (not contiguous; the nine streams were filed in parallel), all pending 2026-11-11: δ 9127 (low), β 9133, α1 9136, α2 9139, γ 9145, ε 9149 (low), ζ 9154 | none (no high leaves) | none held; no high leaf, so no 6698 edge. Chain δ → β → α1 → α2 → γ → ε → ζ (R54); ζ 9154 mints INV-17 `merge-gate-unmeasured-typed` and is the integration gate. β, ε → 7720; γ → 3918; 4767, 6242 → β; δ β α1 α2 ε parked on 8528, so δ (and S3 ε2 through it) waits on 8528 → 8527 → 7720 (R55). R33 (β 9133), R34 (δ 9127), R50 (ε 9149). 6575 and 6576 cancelled and 6242 amended by the clerk pass 2026-11-11. Lead action open: file the reify-side change removing the `speculation_probe` stanza from reify's live yaml before β deploys (β's reify external dependency awaits that id). |
| S8 harness-composition-root-and-slot-registry (wave 2; Streams table: spawn, wave 2, high-risk; upstream S5, S9, S3) | plans/review-all-s8-harness-composition-root-and-slot-registry-prd.md | 22 leaves, ids 9212–9236 (not contiguous; the nine streams were filed in parallel), all pending 2026-11-11: η 9212, α 9216, β0 9217, β 9218, δ 9219 (high), γ 9220 (low), ε1a 9221, ε1b 9222, ε2a 9223 (high), ε2b 9224 (high), ε3 9225, ζ1a 9226, ζ1b 9227, ζ2 9228, θ 9229, ι 9230, ι2 9231, κ1 9232 (low), κ2 9233, μ1 9234 (low), μ2 9235 (low), ω 9236 | 6698 → 9219, 9223, 9224 | none held. fk-b01dac97e6e9 is `fixed:730794ca3b` (task 4686, `PauseClass`; the residual memory/disk divergence is owned by live task 4691); fk-5cad48d24b6f's legality half was a false premise at as_of, its recovery half lands in δ 9219. R35 (θ 9229), R36 (ι 9230, ι2 9231), R37 (η 9212, δ 9219), R38 (β 9218 → 8270), R57 (β0 9217). 4651 → ε2a 9223, and 4651's parking edge on 5040 removed. 3566 cancelled and 5822 amended by the clerk pass 2026-11-11. Lead action open: the eight yaml edits after 9212 deploys (R37). The edge 5153 → α 9216 is NOT wired (5153 has no dependency on 2026-11-11) and awaits Leo's yes. Integration gate / signal leaf ω 9236. No invariant. |
| S9 workflow-phase-spine-and-recovery-ledger (wave 2; Streams table: spawn, wave 2, high-risk; upstream S10, S4) | plans/review-all-s9-workflow-phase-spine-and-recovery-ledger-prd.md | 21 leaves, ids 9164–9289 (not contiguous; the nine streams were filed in parallel), all pending 2026-11-11: π 9164 (high), α 9170 (high), β 9175 (high), γ 9178, δ 9184 (high), ε 9190 (high), ζ1 9196, ζ2 9202, η 9207 (high), θ 9213, ι 9239, κ 9241 (high), λ1 9246 (high), λ2 9249 (high), μ1 9253, μ2 9279, μ3 9281, ν 9284 (high), ξ 9287, ο 9288, ω 9289 | 6698 → 9164, 9170, 9175, 9184, 9190, 9207, 9241, 9246, 9249, 9284 | none held. π 9164 (the L0 settle check script, R40) is high and first; ε 9190 depends on it. NOT FILED: the 14-day L0 check, filed by the lead once 9164 is on main (R40, R58); the S9-5 periodic landed-journal pass is a program follow-up (R39). Existing-task edges include 7052 → α, 7054 → β, κ → 7054; 7115 cancelled. At `commit_planning` eleven substrate checks (π 9164 `escalation-archive-substrate` among them) were refused as `vacuous_present` (already true on main) and not copied into task metadata, by design; every id is stamped and all producer checks were copied. δ 9184 onward waits on 5040 through S5 (R54). Integration gate / signal leaf ω 9289. No invariant. |
| S10 task-write-contract-metadata-modes-success-signal-and-registry (wave 2; Streams table: spawn, wave 2, high-risk) | plans/review-all-s10-task-write-contract-metadata-modes-success-signal-and-registry-prd.md | 21 leaves, ids 9180–9286 (not contiguous; the nine streams were filed in parallel), all pending 2026-11-11: α 9180 (high), β1 9187 (high), β2 9191 (high), γ 9195 (high), π 9199, δ 9204, ε 9209, ζ 9214, ρ 9237, η 9240 (high), θ 9242 (high), ι 9244, κ1 9248, κ2 9251, λ 9254, μ1 9278, μ2 9280, ν 9282, ξ 9283, ο 9285, ω 9286 (high) | 6698 → 9180, 9187, 9191, 9195, 9240, 9242, 9286 | none held. θ 9242 mints INV-16 `metadata-slices-one-table` (R51). α 9180 → 4867 (hand-carried, R47); ν 9282 → 6236 (R53). R41 (κ1 9248), R42 (δ 9204, ε 9209), R43 (λ 9254), R44 (ζ 9214, ρ 9237). 3260 is amended (clerk pass 2026-11-11) and waits on ο 9285 and ξ 9283 (R48 S10-4). NOT FILED: σ, the 30-day combine-collision check, filed by the lead once ρ 9237 is on main (R44, R58). S14 ξ 6719 amended to import `error_envelope` and 6236 amended to extraction only (clerk pass 2026-11-11). Lead actions open: the three-row `files_to_modify` backfill after δ deploys (R42); the optional low curator-field rename. β1 9187 also carries fk-5c98cc0bb7e0 in metadata only. Integration gate / signal leaf ω 9286. |
| S12 reconciliation-run-lifecycle-and-stage-interface (spawn, wave 3; upstream S14) | — | — | — | not authored — R24 later sitting never held; 21 keys open (R12); its S14 upstream leaves are largely landed |
| S13 recon-sweeps-flag-identity-and-recon-report-state (spawn, wave 3; upstream S14, S12) | — | — | — | not authored — R24 later sitting never held; 20 keys open (R13) |
| S15 fused-memory-mcp-server-boundary (spawn, wave 3) | — | — | — | not authored — R24 later sitting never held; 12 keys open (R14); S22 consumes its read model |
| S16 orchestrator-config-shape-defaults-and-routing (spawn, wave 3) | — | — | — | not authored — R24 later sitting never held; 11 keys open; owns R22 invariant 5 (R15); S25 λ 6766 (shared.route_rejection) was filed for it to consume |
| S17 scheduler-seams-and-tick-state (spawn, wave 3; upstream S14, S8) | — | — | — | not authored — R24 later sitting never held; 13 keys open (R16) |
| S19 package-layering-and-private-seams (spawn, wave 3; upstream S14) | — | — | — | not authored — R24 later sitting never held; 12 keys open; owns R22 invariant 3 (R17) |
| S20 cli-invocation-and-usage-gate (spawn, wave 3; upstream S16) | — | — | — | not authored — R24 later sitting never held; 19 keys open (R18); S25 ι 6764 is held on its C3 leaf (`shared.capability_manifest` sidecar_for / tracked_sidecar_paths) and 7724 waits behind it |
| S22 dashboard-readers-and-aggregation (agent, wave 1 in the Streams table; not in R21/R24's wave-1 six; upstream S15, S14) | — | — | — | not authored — omitted from the wave-1 `/team` and no later sitting held; 22 keys open (R19); its S15 upstream was never authored either |
| S24 fleet-watchdog-and-redeploy-coordination (spawn, wave 3; upstream S14) | — | — | — | not authored — R24 later sitting never held; 9 keys open (R20) |
| S26 memory-service-write-path-and-graph-client (wave 2; Streams table: spawn, wave 2, high-risk) | plans/review-all-s26-memory-service-write-path-and-graph-client-prd.md | 23 leaves, ids 9143–9252 (not contiguous; the nine streams were filed in parallel), all pending 2026-11-11: α 9143, β 9147, γ0 9151, γ 9158, δ 9161, ε 9168 (high), ζ 9172, η 9174, θ 9179 (high), θ2 9183, ι 9185, κ 9189, λ 9194 (high), σ 9198 (high), μ 9203, ν 9208, ξ 9215 (high), ο 9238, ρ 9243, τ 9245, υ 9247, π 9250, φ 9252 (low) | 6698 → 9168, 9179, 9194, 9198, 9215 | none held. σ 9198 is the deterministic, always-escalating human design-session gate (R46; `files: []`), pointing at `plans/fm-cross-process-identity-exclusion-session-brief.md`, which goes into the wave-2 commit; it depends on λ 9194 only. R45 (ρ 9243, τ 9245, υ 9247, φ 9252), R56 (θ 9179 → 7743). 5612 cancelled into ξ 9215 by the clerk pass 2026-11-11 (its branch `task/5612`, 54 commits ahead of main, is set aside under the S26-6 ruling); 6293 → σ, ν; 6730 does not depend on σ. ε 9168 also carries fk-ec3de5b6cf2b (6601) in metadata. At `commit_planning` seven substrate checks (β 9147 `mem0-reading-exists` among them) were refused as `vacuous_present` (already true on main) and not copied into task metadata; every id is stamped. Integration gate π 9250. No invariant. |
