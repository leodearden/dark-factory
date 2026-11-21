# Review — dark_factory — review-dark_factory-20261009

## Method

```yaml
run_id: review-dark_factory-20261009
as_of_sha: 6551b75aba6b1885a2aea4bfc46ac82446fba4d4
since: none
evidence:
  changed_files: null
  phase2_files: 671
  phase2_files_note: 'full mode: every tracked src file of shared, escalation, orchestrator, fused-memory, dashboard, sampler
    and scripts/ (690 domain src files minus 19 cockpit); 134 modules deep-read across 8 judging seats, the rest reached by
    grep, AST scans and the import graph'
  carried_findings: 0
  briefing: review/briefing.yaml (header comment 'Last updated 2026-06-23'; no last_updated key)
  metrics_snapshot: none committed under plans/quality-metrics/ at as_of; computed in-run with scripts/quality_metrics_snapshot.py
    (2452/2452 files, complete=true; import graph 2054 edges, 6 reach-backs, 480 deferred imports, 0 cycles) as uncommitted
    context only
  task_store: read-only forensic sqlite on /home/leo/src/dark-factory/.taskmaster/tasks/tasks.db (deferred-task census, owner
    lookups)
verification:
  confirmed: 151
  weakened: 8
  refuted: 0
  unverified: 0
cost:
  agents: 10
  subagent_tokens: 2971783
  wall_clock_min: 133
inputs_consumed:
- legacy:bug-hotspot-survey-2026-07-06-full-findings.json
- codebook@6551b75aba6b1885a2aea4bfc46ac82446fba4d4
extra:
  project_id: dark_factory
  mode: full
  full_reason: --since none
  scope: full
  phases_run:
  - 1
  - 2
  findings_only: true
  launched_from: null
  briefing_defects:
  - '1: briefing.yaml: top-level: no `last_updated` key (only a header comment `# Last updated: 2026-06-23T12:00:00Z`)'
  - '2: subprojects: pyproject [tool.uv.workspace].members has `cockpit` but briefing subprojects lacks it (cockpit is out
    of /review scope this run; decide: add or declare exempt). Other six match both ways'
  - '3: subprojects.fused-memory.known_gaps[0]: tracking: null and legacy `tracking:` key (needs accepted_by) - no e2e FalkorDB/Qdrant
    tests'
  - '3: subprojects.fused-memory.known_gaps[1]: tracking: null (cap detection by text-pattern)'
  - '3: subprojects.fused-memory.known_gaps[2]: legacy tracking: ''1877'' -> task 1877 is DONE; marker TODO(task-1139-gc)
    and _KNOWN_BUG_1139 guard no longer present at pinned tree (gap is stale; remove entry)'
  - '3: subprojects.orchestrator.known_gaps[0]: tracking: null (bwrap/Bun sandbox)'
  - '3: subprojects.orchestrator.known_gaps[1]: legacy tracking: ''1878'' -> task 1878 DONE; no TODO(train, beta2) found in
    git_ops.py at pin (stale gap; verify whether stacked-reuse behaviour is still unfixed)'
  - '3: subprojects.orchestrator.known_gaps[2]: legacy tracking: ''1876'' -> task 1876 DONE; TODO(gamma) gone from verify_runner.py
    (stale; remove)'
  - '3: subprojects.shared.known_gaps[0]: tracking: null (no cross-process usage gate coordination)'
  - '3: subprojects.dashboard.known_gaps[0]: legacy tracking: ''1868'' -> task 1868 DONE (park-stack visibility batch 1868-1874
    all done); gap likely resolved/stale'
  - '3: subprojects.escalation.known_gaps[0]: legacy tracking: ''1879'' -> task 1879 DONE; queue.py TODOs gone (stale; check
    whether archive-scan cost still unbounded, else remove)'
  - '3: subprojects.escalation.known_gaps[1]: tracking: null (duck-typed harness._merge_worker coupling)'
  - '3: subprojects.sampler.known_gaps[0]: tracking: null (duplicated load-samples schema, no sync check)'
  - '3: all known_gaps: none use `accepted_by`; 8 of 12 have tracking: null (defect), 4 point at done tasks; no `in-flight`
    entries; no cancelled-with-x_acceptance_reason cases'
  - '4: conventions[0].why: cites live markers at verify_runner.py:531, git_ops.py:915, queue.py:240/:927, task_knowledge_sync
    1139-gc - all four absent at pin; tasks 1876-1879 all done. Statement ''four live app-code markers'' is false; real live
    marker is harness.py TODO(task sigma) (uncited)'
  - '4: conventions[1].why (deferred-task invariant): says ''Both current deferred tasks (1147, 1853)'' - 1147 is done, 1853
    is cancelled; there are now 336 deferred tasks'
  - '4: subprojects.dashboard.known_gaps[0].why: cites tasks 1868-1874, 1865, 1867 - all exist (done); tracking-claim to done
    task (see 3)'
  - '4: other cited ids 1139,1754,1755,1758,1759,1799,1803-1819,1854,1855,3802,3830 exist (all done; cited as history, fine).
    Line-pin paths in prose (verify_runner.py:531 etc.) are stale line pins. Paths/symbols other than those in known_gaps/conventions[0]
    were not exhaustively resolved by this seat (only task ids and the TODO sites checked)'
  - '6: conventions[1]: invariant violated at scale - 336 deferred tasks, ~333 with no x_deferral/milestone/deferred_watch
    metadata; see deferred_summary'
  - 'staleness: subprojects.sampler purpose/key_scenarios: ''9 load metrics'' and a ''24-hour rolling window (per-tick prune
    + daily VACUUM)''; sampler/store.py retains 30 days, prunes at most once per 24h, VACUUMs only above a 10% free-page fraction,
    and writes 11 + 2 per cgroup-leaf metrics (29 distinct in the last 30 s on this host; live DB 8.17M rows / 934 MB spanning
    19.4 days)'
  - 'staleness: subprojects.fused-memory.what_working_means: thread band ''~33-40 Python threads''; live thread_monitor plateaus
    at 48-51 (3.5 h after restart) on this 32-core host'
  - 'staleness: subprojects.fused-memory.what_working_means: ''get_status returns healthy for all backends (Graphiti, Mem0,
    Taskmaster)''; get_status reports graphiti and mem0 only (the task store is the SQLite backend)'
  - 'staleness: subprojects.fused-memory.key_scenarios[1]: ''dependent tasks get memory_hints''; targeted.py writes hints
    only for newly blocked tasks (FMR seat)'
  - 'staleness: subprojects.orchestrator.known_gaps[1] (1878, train beta2 reuse rebase): closed in tree — create_worktree''s
    reuse branch resolves the predecessor tip for order>0 (SVG seat)'
  - 'staleness: subprojects.escalation.known_gaps[0] (1879): TODOs gone, memo/negative cache/.seq counters landed; the live
    remainder is get_by_task archive cost (ESC-5)'
  - 'staleness: subprojects.dashboard.purpose ''Read-only monitoring web UI'': app.py proxies six write tools (DASH-17)'
  - 'staleness: subprojects.shared.key_decisions: ''one implementation (~1376 lines in shared)'' — shared/usage_gate.py is
    3,052 lines'
  trigger_chain:
    form: none
    task_ids: []
    superseded: []
    reason: '--findings-only: the parent /review-all run triages these findings and files the one chain'
  cost_note: 8 opus judging seats (high effort) + 2 sonnet mechanical seats (low effort); the memory-recon seat additionally
    forked 3 sub-readers whose tokens may not be in the reported total
  excluded: cockpit/ (workspace member with no briefing subproject — reported as a briefing defect, not mapped, per the review-all
    overlay); briefing exclude/global_exclude lists
  not_deep_read:
    scheduler-verify-gitops: suffix_graph, flake_ledger/flake_recorder/chronic_flake, proc_supervision, mcp_lifecycle, cli,
      routing, verify_cancel, verify_classify, b3_gate, mcp/verdict_tools, mcp/markup_sink, evals/**
    shared-legibility: mcp_markup_middleware, memory_eval_limits, delivered_check_polarity, uuid_prefix_guard, toolcall_markup,
      transcript_archive, ratchet, governed_exceptions, proc_group, psi, locking, safe_io, config_dir; legibility digest,
      sampling, nightly.run_nightly, session_runner
    escalation-scripts: pins, disposition, shadow_ruling, action_effects, declared_pins, render_systemd_unit, check_*_unit_parity,
      _task_db_scan, tasks_db_schema
    server-task-layer: server/recon_report.py (3,932 lines)
  codebook_notes:
  - 'unverified-task-premises: its anchor claim ''premise_lint unconsumed'' is refuted at as_of (premise_lint_guard + submit_task
    consume it)'
  - 'one-shot-subagent-contract: largely remediated at as_of (structured verifier outcome, judge infra-failure streak); residual
    kept as FMR-23 (weakened)'
  - 'block-report-misattribution: fix (a) landed (_format_reescalation_detail threads the escalation detail); not re-reported'
  operational_notes:
  - 'fused-memory search returned degraded results during this run: the graphiti store failed with ''Query timed out'' on
    both context searches (mem0 only)'
  - 'smoke side effect: `python -m sampler --help` ignores argv and ran one sampling tick, which DARK_FACTORY_ROOT routed
    to the live data/load-samples.db (one extra row set at ts 1791537126 — equivalent to one timer tick)'
  - 'contract dependency missing at as_of: scripts/check_run_completion.py, the completion-gate predicate named in docs/quality-findings-contract.md
    §11, does not exist — the parent''s trigger chain takes the interim human-gate-only form per skills/_shared/filing-the-trigger-chain.md'
  - 'audit: no .claude/skills/audit/SKILL.md in dark-factory — Step 1 skipped'
  audit:
    audit_skill_present: false
  keys_minted_with: skills/hotspot-survey/scripts/findings_artefact.py::finding_key (interim reference; shared/src/shared/finding_key.py
    absent at as_of)
```

## Review complete: full — review-dark_factory-20261009
as_of 6551b75aba · since none · mode full (findings-only: Phases 1–2; the parent /review-all triages)

### Phase 1
- Tests: 73,839 passed, 0 failed (0 new) — 8 segments, each member independently; 25 skipped, 7 xfailed
- Lint: clean (4 invalid-noqa warnings) · Type-check: clean (0 errors; 1 warning in a fused-memory test)
- Smoke: 9/12 passed, 1 failed (fused-memory thread band — stale band, plateau, no leak shown), 2 not run (side effects / live behaviour)

### Phase 2
- Scope: full — 671 src files in scope, 134 modules deep-read by 8 judging seats; 0 carried findings (no prior contract-shaped report)
- Findings: 26 high · 102 medium · 31 low — by lens: h10 15, h12 14, inv-11 13, tests 13, inv-5 12, h11 12, h14 12, h7 8, h13 8, h3 6, inv-7 6, h6 6, h1 5, comments 5, inv-2 4, inv-8 4, h4 3, h9 3, h2 2, inv-4 2, h5 2, inv-3 1, inv-13 1, inv-6 1, inv-9 1
- By area: fused-memory 54, orchestrator 52, shared 18, dashboard 15, escalation 14, repo 5, sampler 1
- Verification: 151 confirmed · 8 weakened · 0 refuted · 0 unverified
- Fixed since last run: n/a (no prior run)
- Briefing defects: 27 (validate checks 1–6 plus staleness the seats measured)

### Phase 3
Not run (--findings-only). Every finding is `open` / `not-routed`; owners named in statements are leads for the parent's §8 dedup, not dispositions.

### Highest-severity findings

- **DASH-1** `dashboard/src/dashboard/data/merge_halt.py::_probe_one` (inv-5, h11, h3) — The 'probe every escalation URL concurrently' fan-out is hand-rolled four times (inv-5 no-lockstep-duplication; heuristic 11 SPOT; heuristic 3 — one axis, four mechanisms): merge_halt.py::get_merge_halt_status/_probe_one, merge_queue.py::fetch_live_merge_queue…
- **DASH-3** `dashboard/src/dashboard/data/merge_queue.py` (h7, inv-5, h11) — The dashboard reads orchestrator-private runs.db formats by restating the orchestrator's vocabularies as SQL string literals (heuristic 7 — interaction through another module's private storage format; inv-5/heuristic 11 — vocabulary copies; heuristic 12 — enum…
- **DASH-2** `dashboard/src/dashboard/data/metrics.py::fan_out_list_tickets` (inv-11, h10, inv-13) — On the curator-queue scenario a fused-memory outage is indistinguishable from an empty queue (inv-11 no-silent-fail-soft). fan_out_list_tickets passes `offline_result=lambda errs: (0, [])` per root and substitutes (0, []) for any per-root exception, so a root…
- **ESC-2** `escalation/src/escalation/authority.py::l2_auto_close_class` (inv-3, h12, inv-10) — This violates INV-3 (corroborate-before-acting), applied through heuristic 12 (structured data instead of meaningful strings). The one place the L1 auto-watcher identity may close a human-facing L2 above its level ceiling (`resolve_issue`'s level_forbidden bra…
- **ESC-1** `escalation/src/escalation/queue.py::EscalationQueue.set_resolve_callback` (h10, h7, tests) — Heuristic 10 (one mechanism, uniformly enforced) and heuristic 7 (no interaction that depends on an unstated precondition) are both broken here: the thread a loop-affine callback runs on depends on whether the MCP tool that triggered it is spelled `def` or `as…
- **ESC-3** `escalation/src/escalation/queue.py::EscalationQueue.submit` (h10, h9, h11) — This fails heuristic 10 (one home, enforced at more than one point) and heuristic 9 (the policy is not inside the module everyone uses). The ladder invariants live only inside the `create_server` closure `_chokepoint_or_submit`, and nowhere else: born-at-L2 ⇔…
- **FMT-1** `fused-memory/src/fused_memory/middleware/task_interceptor.py::TaskInterceptor._apply_status_transition` (h10, h12, h11) — Heuristic 10 (clear invariants, uniformly enforced: one mechanism, not ad-hoc ifs) fails for the status-transition gate family. TaskInterceptor._apply_status_transition is 595 lines and inlines about ten gates in a hand-ordered sequence: recon_write_policy, sa…
- **FMR-18** `fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py` (tests, h6, h1) — Tests stance + heuristic 6: 11 test files import 47 distinct private names from this 5,666-line module at 144 import sites (_query_stage2_flags 25, _sweep_stale_mem0_flag_markers 19, _sweep_stale_mem0_flag_for_stage2_markers 11, _sweep_stale_persistence_marker…
- **FMT-3** `fused-memory/src/fused_memory/server/tools.py::create_mcp_server` (h3, h10, h9) — Heuristic 3 (orthogonal dimensions, one mechanism per axis) and the cross-module question this seat was asked: the ~35 middleware guard modules do NOT compose through one registration point. The submit_task closure sequences 11 guards inline: lock-charter γ, d…
- **FMT-6** `fused-memory/src/fused_memory/server/tools.py::create_mcp_server` (h10, kind:convention, h7) — Convention 'Task project_root must be an absolute path — empty/relative/None is rejected' and heuristic 10 (enforced redundantly, at construction and at use). The invariant is enforced only by hand-placed `_normalize_project_root(...)` calls at the top of 21 t…
- **FMR-1** `fused-memory/src/fused_memory/services/memory_service.py::MemoryService` (tests, h9, kind:coverage) — Tests stance: test files referencing MemoryService make 709 private-name accesses on service objects covering 39 distinct private names (test_memory_service.py 313, test_referent_repair.py 147, test_referent_verification.py 116, test_referent_queue_threading.p…
- **HWA-3** `orchestrator/src/orchestrator/harness.py::Harness._reconcile_one_stranded` (h11, h3, h2) — Heuristic 11, SPOT. The stranded-task recovery policy no longer has one home. task_ground_truth.py::_RECOVERY is presented as the classification table (TruthReport -> RecoveryAction), but _reconcile_one_stranded (836 lines, 65% prose) rewrites its answer at ni…
- **HWA-2** `orchestrator/src/orchestrator/harness.py::build_train_callback_factory` (inv-7, inv-6, inv-5) — stability_concerns[0], silent strand of non-terminal parked work, plus INV-7 holds-owned-and-bounded. A train member whose delivered_checks are withheld after the train advanced main sits in 'merge-deferred'. Both copies of the recovery edge say in their docst…
- **MLANE-5** `orchestrator/src/orchestrator/merge_lane/worker.py` (tests, h9, h13) — Tests reach deep into the worker's internals, and the production seams are shaped to keep those reaches working (the Tests stance: the seam they reach through is the defect). String-path patches into lane internals remain. 'orchestrator.merge_queue.run_scoped_…
- **MLANE-6** `orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker` (h14, h13, h4) — h14 measurement protocol for worker.py, 22,081 lines (about 11x the 2,000 alarm). Prose first: 12,863 lines are comments or docstrings (58%), 7,534 are code and 1,684 are blank. SpeculativeMergeWorker alone is 13,003 lines, 63% prose. Its four largest methods…
- **MLANE-1** `orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._maybe_coalesce_waiting_singles` (inv-7, inv-6, h5) — This is the HEADLINE strand (stability_concerns[0], inv-7 holds-owned-and-bounded), live in production: dark-factory-orchestrator.yaml sets merge_train_coalesce_enabled: true. Train formation works like this. _maybe_coalesce_waiting_singles resolves every abso…
- **MLANE-7** `orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._verifier_loop` (h2, inv-5, h4) — _verifier_loop is 670 lines with control nesting depth 9 and cognitive complexity 252, the repository's third-highest function and the merge lane's highest. That is up from 245 at the PRD baseline, so per h2's max/total pair the maximum is still rising. It doe…
- **MLANE-2** `orchestrator/src/orchestrator/merge_lane/worker.py::_do_train_merge` (inv-7, h10, inv-11) — A coalesce train can leave the whole merge queue halted with no owner and no escalation, which is the briefing's 'Silent merge-queue halt' concern. When _advance_train ends in any _HALT_ADVANCE_RESULTS code (wip_overlap, pop_conflict, unmerged_state, pop_confl…
- **SVG-4** `orchestrator/src/orchestrator/scheduler.py::Scheduler._compute_delivered_check_cache` (inv-7, inv-4, inv-3) — Two defects at one anchor. (1) INV-7 / INV-4: a dep whose delivered_check returns ERRORED (missing or non-executable script, bad path, timeout; delivered_checks.py::run_delivered_check maps every runner exception and a malformed descriptor to ERRORED) is left…
- **SVG-3** `orchestrator/src/orchestrator/scheduler.py::Scheduler._deps_satisfied` (h10, h1, h3) — Heuristic 10 (invariant not uniformly enforced) and heuristic 1: 'deps satisfied' has one name but three meanings selected by optional arguments. Scheduler._deps_satisfied takes four optional gates (external_status_cache, external_resolver_failed, delivered_ch…
- **SVG-1** `orchestrator/src/orchestrator/verify.py::_build_fallback_config` (inv-5, h11, h1) — INV-5 no-lockstep-duplication / heuristic 11 (SPOT): the per-task verify scope decision still has two engines. verify_plan.py's module docstring claims it 'Unifies the twice-fixed scope decision between scope_module_config and _build_fallback_config behind a s…
- **SVG-7** `orchestrator/src/orchestrator/verify.py::_run_cmd` (tests, h9, h13) — Tests stance: orchestrator.verify is the most internally-reached module in the repo. Its only subprocess seam is the private module function _run_cmd, patched by dotted path 199 times across 19 test files ('orchestrator.verify._run_cmd'); further dotted patch…
- **HWA-1** `orchestrator/src/orchestrator/workflow.py::TaskWorkflow._mark_blocked` (h3, h6, tests) — Heuristic 3, carefully factored orthogonal dimensions of variability: _mark_blocked encodes status-write timing (merge_phase, block_status), escalation routing (skip_escalation, escalate_to_human, category, dedupe_fingerprint, suggested_action) and dry-run spa…
- **ESC-4** `scripts/orchestrator-watchdog.py::main` (inv-4, h10, inv-7) — The orchestrator liveness tier fails INV-4 (storm-escape-required) and heuristic 10, in two ways. (1) It cannot see the failure that matters most. Its only signals are `probe_port` (an `ss -ltn` LISTEN check) and systemd ActiveState. Under socket activation th…
- **SHL-2** `shared/src/shared/cli_invoke.py::invoke_with_cap_retry` (h2, h4, h3) — Heuristic 2 and 4: invoke_with_cap_retry is one 1,016-line function (cognitive 142, the slice maximum) whose gated path is a single `while True: async with usage_gate.invoke_slot()` body holding ten ordered arms (AuthFailed, ModelNotFound, CLI-reject retry, Ze…
- **SHL-6** `shared/src/shared/usage_gate.py::UsageGate` (tests, h10, h9) — Tests stance: UsageGate's tests drive its private state rather than its public seam. Across 40 test files (27 under shared/tests, 7 orchestrator, 4 fused-memory) there are 53 distinct `gate._*` private names, led by gate._accounts (810 occurrences), gate._open…

## Findings

| display_id | key | area | anchor | tags | severity | class | verdict | disposition |
|---|---|---|---|---|---|---|---|---|
| DASH-1 | fk-ad49b80c5554 | dashboard | `dashboard/src/dashboard/data/merge_halt.py::_probe_one` | inv-5, h11, h3, inv-4, kind:defect, kind:cross-module | high | structural | confirmed | open |
| DASH-3 | fk-af4ba730a48a | dashboard | `dashboard/src/dashboard/data/merge_queue.py` | h7, inv-5, h11, h12, inv-11, kind:cross-module | high | structural | confirmed | open |
| DASH-2 | fk-4afbeb684a22 | dashboard | `dashboard/src/dashboard/data/metrics.py::fan_out_list_tickets` | inv-11, h10, inv-13, kind:critical-path, kind:convention | high | structural | confirmed | open |
| ESC-2 | fk-48a76ee9f746 | escalation | `escalation/src/escalation/authority.py::l2_auto_close_class` | inv-3, h12, inv-10, kind:deep-read | high | structural | confirmed | open |
| ESC-1 | fk-07be43f313bc | escalation | `escalation/src/escalation/queue.py::EscalationQueue.set_resolve_callback` | h10, h7, tests, kind:cross-module | high | structural | confirmed | open |
| ESC-3 | fk-0d31f1114911 | escalation | `escalation/src/escalation/queue.py::EscalationQueue.submit` | h10, h9, h11, inv-5, h12, kind:cross-module | high | structural | confirmed | open |
| FMT-1 | fk-d80c1c44bca1 | fused-memory | `fused-memory/src/fused_memory/middleware/task_interceptor.py::TaskInterceptor._apply_status_transition` | h10, h12, h11, h4, h9, kind:cross-module | high | structural | confirmed | open |
| FMR-18 | fk-90f374fcce21 | fused-memory | `fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py` | tests, h6, h1, h9, kind:deep-read | high | structural | confirmed | open |
| FMT-3 | fk-1509bcd45e2a | fused-memory | `fused-memory/src/fused_memory/server/tools.py::create_mcp_server` | h3, h10, h9, h11, kind:cross-module | high | structural | confirmed | open |
| FMT-6 | fk-330a60eef271 | fused-memory | `fused-memory/src/fused_memory/server/tools.py::create_mcp_server` | h10, kind:convention, h7, inv-11 | high | structural | confirmed | open |
| FMR-1 | fk-50cc281018fb | fused-memory | `fused-memory/src/fused_memory/services/memory_service.py::MemoryService` | tests, h9, kind:coverage | high | structural | confirmed | open |
| HWA-3 | fk-17b789fe3ce6 | orchestrator | `orchestrator/src/orchestrator/harness.py::Harness._reconcile_one_stranded` | h11, h3, h2, comments, kind:deep-read | high | structural | confirmed | open |
| HWA-2 | fk-1b573c2428c2 | orchestrator | `orchestrator/src/orchestrator/harness.py::build_train_callback_factory` | inv-7, inv-6, inv-5, h11, kind:invariant | high | structural | confirmed | open |
| MLANE-5 | fk-e7deb8bcfa95 | orchestrator | `orchestrator/src/orchestrator/merge_lane/worker.py` | tests, h9, h13, kind:coverage | high | structural | confirmed | open |
| MLANE-6 | fk-e521254a5b71 | orchestrator | `orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker` | h14, h13, h4, h6, comments, kind:deep-read | high | structural | confirmed | open |
| MLANE-1 | fk-e8a03e4ec067 | orchestrator | `orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._maybe_coalesce_waiting_singles` | inv-7, inv-6, h5, kind:critical-path | high | structural | confirmed | open |
| MLANE-7 | fk-1432a52e4753 | orchestrator | `orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._verifier_loop` | h2, inv-5, h4, h5, kind:deep-read | high | structural | confirmed | open |
| MLANE-2 | fk-ba531799522b | orchestrator | `orchestrator/src/orchestrator/merge_lane/worker.py::_do_train_merge` | inv-7, h10, inv-11, kind:critical-path | high | structural | confirmed | open |
| SVG-4 | fk-6c103745c784 | orchestrator | `orchestrator/src/orchestrator/scheduler.py::Scheduler._compute_delivered_check_cache` | inv-7, inv-4, inv-3, h4, comments, kind:critical-path | high | structural | confirmed | open |
| SVG-3 | fk-4b35b1adea0d | orchestrator | `orchestrator/src/orchestrator/scheduler.py::Scheduler._deps_satisfied` | h10, h1, h3, h7, inv-7, kind:critical-path | high | structural | confirmed | open |
| SVG-1 | fk-fac2cb6af858 | orchestrator | `orchestrator/src/orchestrator/verify.py::_build_fallback_config` | inv-5, h11, h1, tests, comments, kind:cross-module | high | structural | confirmed | open |
| SVG-7 | fk-3e806baf4c6e | orchestrator | `orchestrator/src/orchestrator/verify.py::_run_cmd` | tests, h9, h13, kind:coverage | high | structural | confirmed | open |
| HWA-1 | fk-b8efaa973162 | orchestrator | `orchestrator/src/orchestrator/workflow.py::TaskWorkflow._mark_blocked` | h3, h6, tests, kind:defect, kind:critical-path | high | structural | confirmed | open |
| ESC-4 | fk-1cc71283157c | repo | `scripts/orchestrator-watchdog.py::main` | inv-4, h10, inv-7, kind:critical-path | high | structural | confirmed | open |
| SHL-2 | fk-2a5d97aeb64d | shared | `shared/src/shared/cli_invoke.py::invoke_with_cap_retry` | h2, h4, h3, h11, h7, comments, kind:deep-read | high | structural | confirmed | open |
| SHL-6 | fk-25c96d016239 | shared | `shared/src/shared/usage_gate.py::UsageGate` | tests, h10, h9, kind:coverage | high | structural | confirmed | open |
| DASH-9 | fk-ce66d4da9b0f | dashboard | `dashboard/src/dashboard/app.py` | h6, h9, h12, comments, kind:deep-read | medium | structural | confirmed | open |
| DASH-10 | fk-166cacf84a18 | dashboard | `dashboard/src/dashboard/app.py::_probe_mcp_fanout` | tests, h5, h12, h10, comments, kind:coverage | medium | structural | confirmed | open |
| DASH-5 | fk-c893f7fb4647 | dashboard | `dashboard/src/dashboard/app.py::api_curator_cancel` | inv-5, h11, kind:defect, kind:deep-read | medium | mechanical | confirmed | open |
| DASH-7 | fk-5e542936151c | dashboard | `dashboard/src/dashboard/data/active_tasks.py::_project_label` | h11, inv-5, h1, h12, h13, kind:cross-module | medium | structural | confirmed | open |
| DASH-6 | fk-d21f64525e52 | dashboard | `dashboard/src/dashboard/data/active_tasks.py::task_uid` | h12, h11, kind:cross-module | medium | structural | confirmed | open |
| DASH-8 | fk-9979f1b771de | dashboard | `dashboard/src/dashboard/data/db.py::with_db` | inv-11, h10, h3, kind:convention | medium | structural | confirmed | open |
| DASH-11 | fk-c51423b7b7e5 | dashboard | `dashboard/src/dashboard/data/mcp_fanout.py` | h6, h13, comments, tests, h1, kind:deep-read | medium | structural | confirmed | open |
| DASH-4 | fk-d0a7ebc4a4f1 | dashboard | `dashboard/src/dashboard/data/orchestrator.py::discover_orchestrators` | inv-5, h11, h7, h12, kind:cross-module | medium | structural | confirmed | open |
| DASH-12 | fk-6aee6c49fa2e | dashboard | `dashboard/src/dashboard/data/scheduler.py::collect_scheduler_state` | h4, h2, h8, comments, kind:deep-read | medium | structural | confirmed | open |
| ESC-12 | fk-ce2dc99e0305 | escalation | `escalation/src/escalation/archive.py::prune_archive` | h1, inv-11, kind:cross-module | medium | structural | confirmed | open |
| ESC-9 | fk-638cca2f65bb | escalation | `escalation/src/escalation/dedupe.py::summary_dedupe_key` | h12, h3, kind:deep-read | medium | structural | confirmed | open |
| ESC-10 | fk-479375564c2f | escalation | `escalation/src/escalation/models.py::EvidenceEntry` | inv-2, h12, h10, kind:invariant | medium | structural | confirmed | open |
| ESC-5 | fk-e5d81cc1f1d4 | escalation | `escalation/src/escalation/queue.py::EscalationQueue.get_by_task` | inv-8, h9, h10, kind:deep-read | medium | structural | confirmed | open |
| ESC-11 | fk-43f431f697a3 | escalation | `escalation/src/escalation/server.py` | h14, comments, h13, h9, h4, kind:deep-read | medium | structural | confirmed | open |
| COORD-1 | fk-e903cc072b15 | escalation | `escalation/src/escalation/server.py` | h9, h13, h7, kind:cross-module | medium | structural | confirmed | open |
| ESC-8 | fk-60b7c33ea0ad | escalation | `escalation/src/escalation/server.py::CATEGORIES` | h12, h11, inv-1, kind:convention | medium | structural | confirmed | open |
| HWA-9 | fk-f8e9e0da3786 | escalation | `escalation/src/escalation/server.py::_get_merge_worker` | h7, h9, h12, kind:cross-module | medium | structural | confirmed | open |
| ESC-6 | fk-66746efddd4d | escalation | `escalation/src/escalation/server.py::_get_terminal_retention` | inv-13, inv-3, h7, kind:critical-path | medium | structural | confirmed | open |
| ESC-7 | fk-bc64099a1cd0 | escalation | `escalation/src/escalation/server.py::create_server` | inv-5, h11, kind:cross-module | medium | structural | confirmed | open |
| FMR-8 | fk-62af616b8470 | fused-memory | `fused-memory/src/fused_memory/backends/graphiti_client.py::GraphitiBackend` | h10, h7, kind:cross-module | medium | structural | confirmed | open |
| FMT-7 | fk-c41aa02e0779 | fused-memory | `fused-memory/src/fused_memory/backends/sqlite_task_backend.py::SqliteTaskBackend.get_statuses_fresh` | inv-11, h10, h12, kind:cross-module | medium | structural | confirmed | open |
| FMR-31 | fk-709d4d84ee37 | fused-memory | `fused-memory/src/fused_memory/config/schema.py::ReconciliationConfig` | h12, h10, h3, kind:cross-module | medium | structural | confirmed | open |
| FMT-11 | fk-51f623a85eee | fused-memory | `fused-memory/src/fused_memory/middleware/operational_routing_guard.py::inject_operational_routing` | h13, h9, h7, kind:cross-module | medium | mechanical | confirmed | open |
| FMT-9 | fk-b259ba3af605 | fused-memory | `fused-memory/src/fused_memory/middleware/task_curator.py::TaskCurator._build_corpus` | inv-8, inv-11, h1, kind:deep-read | medium | structural | confirmed | open |
| FMT-8 | fk-03adb94005f4 | fused-memory | `fused-memory/src/fused_memory/middleware/task_curator.py::TaskCurator.curate_batch_prepared` | h11, h1, inv-5, h4, h2, kind:deep-read | medium | structural | weakened | open |
| FMT-10 | fk-da7758b20d07 | fused-memory | `fused-memory/src/fused_memory/middleware/task_curator.py::_LazyRegistry` | inv-11, inv-4, h10, kind:deep-read | medium | mechanical | confirmed | open |
| FMT-15 | fk-e3e99447e11e | fused-memory | `fused-memory/src/fused_memory/middleware/task_interceptor.py` | h14, h6, h9, h13, h4 | medium | structural | confirmed | open |
| FMT-13 | fk-85e1293e0aa4 | fused-memory | `fused-memory/src/fused_memory/middleware/task_interceptor.py::TaskInterceptor.__init__` | h10, h7, h3, h5, inv-11, kind:deep-read | medium | structural | confirmed | open |
| FMT-12 | fk-06ab2d3e9148 | fused-memory | `fused-memory/src/fused_memory/middleware/task_interceptor.py::TaskInterceptor._is_gate_metadata` | h11, h1, h12, comments, kind:cross-module | medium | structural | confirmed | open |
| FMT-2 | fk-b2f2dcfc1d0e | fused-memory | `fused-memory/src/fused_memory/middleware/task_interceptor.py::_validate_done_provenance` | h11, h10, h12, inv-5, h3, kind:deep-read | medium | mechanical | confirmed | open |
| FMR-28 | fk-4c8d19c64d99 | fused-memory | `fused-memory/src/fused_memory/reconciliation/event_journal.py::EventJournal` | inv-8, h1, inv-5, h11, kind:cross-module | medium | structural | confirmed | open |
| FMR-13 | fk-140888e80723 | fused-memory | `fused-memory/src/fused_memory/reconciliation/flag_dedup.py::filter_false_absence_flags` | h12, h10, inv-11, kind:deep-read | medium | structural | confirmed | open |
| FMR-10 | fk-a88506a1e327 | fused-memory | `fused-memory/src/fused_memory/reconciliation/harness.py` | h14, comments, h13, h6, kind:deep-read | medium | structural | confirmed | open |
| FMR-14 | fk-0485729893fe | fused-memory | `fused-memory/src/fused_memory/reconciliation/harness.py::BacklogIterator` | h7, inv-11, h1, kind:defect | medium | structural | confirmed | open |
| FMR-16 | fk-16baf39b1db6 | fused-memory | `fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness` | tests, h9, kind:coverage | medium | structural | confirmed | open |
| FMR-15 | fk-147a9951edc7 | fused-memory | `fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._replay_deferred_writes` | inv-7, inv-11, h11, inv-5, kind:critical-path | medium | structural | confirmed | open |
| FMR-11 | fk-2571f7726184 | fused-memory | `fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._run_remediation_pass` | h11, inv-5, h4, h2, kind:deep-read | medium | structural | confirmed | open |
| FMR-29 | fk-20cb93d17ff3 | fused-memory | `fused-memory/src/fused_memory/reconciliation/prompts/__init__.py::render_recon_report_tool_guidance` | h13, h9, inv-5, inv-11, comments, kind:cross-module | medium | structural | confirmed | open |
| FMR-27 | fk-d2c7f1b902ac | fused-memory | `fused-memory/src/fused_memory/reconciliation/stages/base.py::_find_fused_memory_server` | h11, kind:defect, inv-11, kind:deep-read | medium | structural | confirmed | open |
| FMR-20 | fk-63b65973bd33 | fused-memory | `fused-memory/src/fused_memory/reconciliation/stages/memory_consolidator.py::MemoryConsolidator` | h7, h5, h8, kind:cross-module | medium | structural | confirmed | open |
| FMR-12 | fk-37356472614a | fused-memory | `fused-memory/src/fused_memory/reconciliation/stages/memory_consolidator.py::MemoryConsolidator.run` | h3, h4, h6, inv-11, h1, comments, kind:critical-path | medium | structural | confirmed | open |
| COORD-9 | fk-875cca636438 | fused-memory | `fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py::_queue_briefing_refresh_tasks` | h10, h9, kind:convention, kind:cross-module | medium | structural | confirmed | open |
| FMR-19 | fk-fa1fd33c7a9e | fused-memory | `fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py::_sweep_stale_mem0_pool` | inv-11, h4, comments, h2, kind:deep-read | medium | structural | confirmed | open |
| FMR-30 | fk-cdb8c20a3d0f | fused-memory | `fused-memory/src/fused_memory/reconciliation/stale_status_snapshot_edge_sweep.py` | h12, comments, kind:deep-read | medium | structural | confirmed | open |
| FMR-22 | fk-05443fb78da6 | fused-memory | `fused-memory/src/fused_memory/reconciliation/targeted.py::TargetedReconciler` | h10, h7, h13, kind:cross-module | medium | structural | confirmed | open |
| FMR-21 | fk-2bb6c913d747 | fused-memory | `fused-memory/src/fused_memory/reconciliation/targeted.py::TargetedReconciler._on_task_done` | inv-11, inv-8, h1, kind:critical-path | medium | structural | confirmed | open |
| FMR-9 | fk-b7502511f063 | fused-memory | `fused-memory/src/fused_memory/routing/classifier.py::WriteClassifier` | inv-11, inv-4, kind:critical-path | medium | mechanical | confirmed | open |
| FMT-17 | fk-b3d95cdaa0d3 | fused-memory | `fused-memory/src/fused_memory/server/main.py` | tests, h5, h4, h13, h6 | medium | structural | confirmed | open |
| FMT-14 | fk-9214f842ca92 | fused-memory | `fused-memory/src/fused_memory/server/tools.py` | h14, h9, h13, h4, h2, tests, comments | medium | structural | confirmed | open |
| FMT-5 | fk-f5e16692e170 | fused-memory | `fused-memory/src/fused_memory/server/tools.py::_OVERRIDE_SCHEMA` | h11, inv-5, h9, inv-9, kind:cross-module | medium | structural | confirmed | open |
| FMT-4 | fk-75fa713ba3d5 | fused-memory | `fused-memory/src/fused_memory/server/tools.py::_resolve_identity` | h12, h11, h10, kind:cross-module | medium | structural | confirmed | open |
| FMT-16 | fk-009fbbf855e7 | fused-memory | `fused-memory/src/fused_memory/server/tools.py::create_mcp_server` | comments, h1, inv-9, kind:deep-read | medium | structural | confirmed | open |
| FMR-5 | fk-343c4a42be6b | fused-memory | `fused-memory/src/fused_memory/services/memory_service.py` | h14, h13, comments, kind:deep-read | medium | structural | confirmed | open |
| FMR-3 | fk-bdc0427c8971 | fused-memory | `fused-memory/src/fused_memory/services/memory_service.py::MemoryService._execute_durable_write` | h12, inv-5, h11, kind:cross-module | medium | structural | confirmed | open |
| FMR-4 | fk-41eaf65cc4d0 | fused-memory | `fused-memory/src/fused_memory/services/memory_service.py::MemoryService._verify_episode_referents` | comments, h4, h2, kind:deep-read | medium | structural | confirmed | open |
| FMR-2 | fk-b30cc4fa2899 | fused-memory | `fused-memory/src/fused_memory/services/memory_service.py::MemoryService.add_memory` | h1, h12, inv-11, kind:critical-path | medium | structural | confirmed | open |
| FMR-26 | fk-f01bf7bc23f9 | fused-memory | `fused-memory/src/fused_memory/services/topic_anchor.py::_CHILD_KINDS` | inv-5, h11, comments, h13, kind:cross-module | medium | structural | confirmed | open |
| HWA-15 | fk-0ee68c0948ab | orchestrator | `orchestrator/src/orchestrator/deterministic_runner.py::DeterministicRunner._file_infra_issue_and_block` | inv-6, inv-5, h10, kind:invariant | medium | structural | confirmed | open |
| HWA-14 | fk-ebce5e23048a | orchestrator | `orchestrator/src/orchestrator/deterministic_runner.py::DeterministicRunner.run` | h12, h4, h3, comments, kind:deep-read | medium | structural | confirmed | open |
| HWA-16 | fk-e43472e7d5d7 | orchestrator | `orchestrator/src/orchestrator/dry_run_unblock.py::_build_entry` | h12, h11, kind:cross-module | medium | structural | confirmed | open |
| SVG-13 | fk-f1a7f57fa58e | orchestrator | `orchestrator/src/orchestrator/event_store.py::EventStore.emit` | inv-8, kind:deep-read | medium | structural | confirmed | open |
| SVG-9 | fk-c00c5f01be4c | orchestrator | `orchestrator/src/orchestrator/git_ops.py` | h14, h6, h9, h13, comments, kind:deep-read | medium | structural | confirmed | open |
| SVG-6 | fk-3ee93d431321 | orchestrator | `orchestrator/src/orchestrator/git_ops.py::GitOps.advance_main` | h7, h5, h8, kind:cross-module | medium | structural | confirmed | open |
| SVG-5 | fk-739af676767e | orchestrator | `orchestrator/src/orchestrator/git_ops.py::GitOps.commit` | h10, h1, comments, inv-9, kind:convention | medium | structural | weakened | open |
| SVG-8 | fk-3a4cc6298c02 | orchestrator | `orchestrator/src/orchestrator/git_ops.py::_run` | tests, h1, h9, h13, kind:cross-module | medium | structural | confirmed | open |
| HWA-17 | fk-b1c6965066ed | orchestrator | `orchestrator/src/orchestrator/harness.py` | h14, h13, comments, kind:deep-read | medium | structural | confirmed | open |
| HWA-8 | fk-d470c7828682 | orchestrator | `orchestrator/src/orchestrator/harness.py::Harness` | tests, h9, h6, kind:coverage | medium | structural | confirmed | open |
| HWA-12 | fk-0fee68ae22d6 | orchestrator | `orchestrator/src/orchestrator/harness.py::Harness._recover_crashed_tasks` | h7, inv-5, h1, h2, kind:critical-path | medium | structural | confirmed | open |
| HWA-13 | fk-69b77cb9d60f | orchestrator | `orchestrator/src/orchestrator/harness.py::Harness._start_escalation_server` | inv-11, h13, h10, kind:deep-read | medium | mechanical | confirmed | open |
| MLANE-9 | fk-689227abfab1 | orchestrator | `orchestrator/src/orchestrator/merge_lane/gates.py::_check_plan_files_touched_in_branch` | inv-11, h1, inv-2, kind:defect | medium | structural | confirmed | open |
| MLANE-3 | fk-80a790e6f793 | orchestrator | `orchestrator/src/orchestrator/merge_lane/types.py::GroupMergeRequest` | h3, h12, h11, kind:cross-module | medium | structural | confirmed | open |
| MLANE-11 | fk-e768d50f8363 | orchestrator | `orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker.__init__` | comments, h13, kind:deep-read | medium | structural | confirmed | open |
| MLANE-14 | fk-83625b943aa5 | orchestrator | `orchestrator/src/orchestrator/merge_lane/worker.py::_WipHaltMixin` | h10, h6, inv-7, h3, kind:deep-read | medium | structural | confirmed | open |
| MLANE-8 | fk-201d70aaf409 | orchestrator | `orchestrator/src/orchestrator/merge_lane/worker.py::classify_and_merge` | h7, h13, h9, kind:deep-read | medium | structural | confirmed | open |
| MLANE-15 | fk-d41caf6b6331 | orchestrator | `orchestrator/src/orchestrator/merge_queue.py` | h13, tests, h9, kind:cross-module | medium | structural | confirmed | open |
| SVG-14 | fk-e22732500172 | orchestrator | `orchestrator/src/orchestrator/session_registry.py::_pid_alive` | inv-5, h11, kind:cross-module | medium | structural | confirmed | open |
| HWA-10 | fk-a1085af3a290 | orchestrator | `orchestrator/src/orchestrator/steward.py::TaskSteward._handle_escalation` | h10, h1, h11, kind:deep-read | medium | structural | confirmed | open |
| SVG-10 | fk-a635d54da4ab | orchestrator | `orchestrator/src/orchestrator/verify.py` | h14, h13, comments, tests, kind:deep-read | medium | structural | confirmed | open |
| SVG-11 | fk-3b67e8761674 | orchestrator | `orchestrator/src/orchestrator/verify.py::VerifyResult` | inv-2, h12, comments, kind:cross-module | medium | structural | confirmed | open |
| SVG-2 | fk-eeab41dce55e | orchestrator | `orchestrator/src/orchestrator/verify.py::_scope_to_keyword` | inv-5, h11, comments, kind:cross-module | medium | mechanical | confirmed | open |
| SVG-15 | fk-0bb03714372a | orchestrator | `orchestrator/src/orchestrator/verify.py::run_verification` | h3, h1, h4, h2, comments, kind:deep-read | medium | structural | confirmed | open |
| MLANE-10 | fk-baa454b81f6f | orchestrator | `orchestrator/src/orchestrator/verify_runner.py::VerifyRunnerPool` | h11, h3, h9, kind:critical-path | medium | structural | confirmed | open |
| HWA-18 | fk-a71254176a81 | orchestrator | `orchestrator/src/orchestrator/workflow.py` | h14, h13, comments, kind:deep-read | medium | structural | confirmed | open |
| HWA-7 | fk-76fee9301ca3 | orchestrator | `orchestrator/src/orchestrator/workflow.py::TaskWorkflow` | tests, h9, h7, kind:coverage | medium | structural | confirmed | open |
| HWA-11 | fk-bab6058b23c8 | orchestrator | `orchestrator/src/orchestrator/workflow.py::TaskWorkflow._await_steward_completion` | h7, h10, inv-5, kind:cross-module | medium | structural | confirmed | open |
| HWA-6 | fk-7d410939a337 | orchestrator | `orchestrator/src/orchestrator/workflow.py::TaskWorkflow._execute_verify_review_loop` | h5, h12, h7, kind:deep-read | medium | structural | confirmed | open |
| HWA-5 | fk-fd91b14b3ead | orchestrator | `orchestrator/src/orchestrator/workflow.py::TaskWorkflow._submit_to_merge_queue` | h12, h3, kind:cross-module | medium | structural | confirmed | open |
| MLANE-4 | fk-75fc1b96098d | orchestrator | `orchestrator/src/orchestrator/workflow.py::_select_train_members` | h9, h13, h1, kind:cross-module | medium | structural | confirmed | open |
| ESC-13 | fk-47a7eb8c020d | repo | `scripts/reclaim-orphaned-worktrees.sh` | inv-7, h11, h12, inv-11, kind:defect | medium | structural | confirmed | open |
| COORD-2 | fk-d1b8c6fbd6c4 | repo | `slug:deferred-task-flip-conditions` | h10, kind:convention | medium | structural | confirmed | open |
| SHL-14 | fk-a0af0f46c1dc | shared | `scripts/legibility/census.py::_find_pending_candidate_id` | h12, inv-5, h11, kind:cross-module | medium | structural | confirmed | open |
| SHL-15 | fk-c9ab2773a5d9 | shared | `scripts/legibility/census.py::run_census` | h4, h6, h12, comments, kind:deep-read | medium | structural | confirmed | open |
| SHL-13 | fk-e3c7486f1f31 | shared | `scripts/legibility/census_trigger.py::extract_done_count` | inv-11, h10, kind:critical-path | medium | mechanical | confirmed | open |
| SHL-8 | fk-3ae32b390a81 | shared | `shared/src/shared/cli_invoke.py` | h14, h13, comments, h6, kind:deep-read | medium | structural | confirmed | open |
| SHL-4 | fk-c75a491d97c5 | shared | `shared/src/shared/cli_invoke.py::_run_subprocess` | inv-2, h1, h10, kind:deep-read | medium | mechanical | confirmed | open |
| SHL-7 | fk-fede65204150 | shared | `shared/src/shared/cli_invoke.py::_run_subprocess` | tests, comments, kind:coverage | medium | structural | confirmed | open |
| SHL-17 | fk-2f4b8655fb3c | shared | `shared/src/shared/cli_invoke.py::invoke_claude_agent` | h10, comments, inv-1, kind:critical-path | medium | structural | confirmed | open |
| SHL-1 | fk-26886ff8653b | shared | `shared/src/shared/cli_invoke.py::invoke_with_cap_retry` | h5, kind:defect, h10, kind:deep-read | medium | mechanical | confirmed | open |
| SHL-18 | fk-de9fe5983ba5 | shared | `shared/src/shared/config_models.py::UsageCapConfig` | inv-11, h11, kind:critical-path | medium | mechanical | confirmed | open |
| SHL-12 | fk-1695d6b02400 | shared | `shared/src/shared/task_statuses.py::TaskStatus` | h12, h11, comments, kind:cross-module | medium | structural | confirmed | open |
| SHL-5 | fk-7bbea5d4e082 | shared | `shared/src/shared/usage_gate.py::UsageGate._run_probe` | h11, inv-5, kind:cross-module | medium | structural | confirmed | open |
| SHL-3 | fk-ae6cd7600d0e | shared | `shared/src/shared/usage_gate.py::UsageGate._transition` | h11, inv-5, h12, kind:cross-module | medium | structural | confirmed | open |
| DASH-15 | fk-895f269d7bb5 | dashboard | `dashboard/src/dashboard/data/load.py` | inv-5, h11, h12, tests, kind:cross-module | low | structural | weakened | open |
| DASH-13 | fk-983fd1bb3c12 | dashboard | `dashboard/src/dashboard/data/memory_evals.py::_build_eval` | h4, h8, h2, comments, kind:deep-read | low | mechanical | confirmed | open |
| DASH-14 | fk-35257037deb7 | dashboard | `dashboard/src/dashboard/data/tasks.py::_walk_pages` | h9, h6, tests, kind:deep-read | low | mechanical | confirmed | open |
| ESC-14 | fk-2748d213f41e | escalation | `escalation/src/escalation/watcher.py::main` | tests, h4, kind:coverage | low | mechanical | confirmed | open |
| FMR-33 | fk-13cdf69a4c0f | fused-memory | `fused-memory/src/fused_memory/backends/graphiti_client.py::_construct_llm_client` | comments, h4, kind:deep-read | low | mechanical | confirmed | open |
| FMR-34 | fk-3db21d545e1e | fused-memory | `fused-memory/src/fused_memory/config/schema.py::TaskmasterConfig` | h1, kind:deep-read | low | mechanical | confirmed | open |
| COORD-6 | fk-7246e801095f | fused-memory | `fused-memory/src/fused_memory/middleware/task_curator.py::_task_metadata_spawned_from` | h6, kind:dead-code | low | mechanical | confirmed | open |
| FMR-25 | fk-c0932f0b7df1 | fused-memory | `fused-memory/src/fused_memory/reconciliation/__init__.py` | h13, h1, kind:cross-module | low | mechanical | weakened | open |
| FMR-17 | fk-0165547376d8 | fused-memory | `fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._fetch_filtered_task_tree` | inv-11, inv-8, kind:critical-path | low | mechanical | weakened | open |
| COORD-3 | fk-101f14d842fa | fused-memory | `fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._recover_one_run` | h10, comments, kind:convention | low | mechanical | confirmed | open |
| FMR-32 | fk-fbbf580e4e1d | fused-memory | `fused-memory/src/fused_memory/reconciliation/policies/__init__.py` | h3, h1, kind:cross-module | low | structural | confirmed | open |
| FMR-24 | fk-9850339b610c | fused-memory | `fused-memory/src/fused_memory/reconciliation/prompts/stage2.py` | inv-5, h1, h11, kind:cross-module | low | structural | confirmed | open |
| FMR-23 | fk-b49ccd852736 | fused-memory | `fused-memory/src/fused_memory/reconciliation/verify.py::CodebaseVerifier` | inv-4, inv-11, kind:invariant | low | mechanical | weakened | open |
| FMR-7 | fk-9b6d91a299b5 | fused-memory | `fused-memory/src/fused_memory/services/__init__.py` | h13, kind:cross-module | low | mechanical | confirmed | open |
| FMR-6 | fk-16d7023811f6 | fused-memory | `fused-memory/src/fused_memory/services/durable_queue.py::DeadLetterEvent` | inv-2, h12, h13, kind:cross-module | low | mechanical | confirmed | open |
| COORD-5 | fk-853e62d7f73d | orchestrator | `orchestrator/src/orchestrator/evals/judge.py::run_tournament` | h6, kind:dead-code | low | mechanical | confirmed | open |
| HWA-19 | fk-43bad4e724da | orchestrator | `orchestrator/src/orchestrator/harness.py::_WATCHER_ESCALATION_HEADERS` | inv-5, h11, h9, comments, kind:cross-module | low | structural | confirmed | open |
| MLANE-13 | fk-a10d37257808 | orchestrator | `orchestrator/src/orchestrator/lane_lifecycle.py::LaneLifecycle` | inv-11, h7, kind:deep-read | low | structural | weakened | open |
| COORD-7 | fk-55e500b51946 | orchestrator | `orchestrator/src/orchestrator/merge_lane/worker.py::_resolve_commit_tree` | h6, kind:dead-code | low | mechanical | confirmed | open |
| MLANE-12 | fk-0de72a91f89a | orchestrator | `orchestrator/src/orchestrator/merge_lane/worker.py::coalesce_or_enqueue_merge_request` | h1, kind:cross-module | low | structural | confirmed | open |
| SVG-12 | fk-0238ff63263c | orchestrator | `orchestrator/src/orchestrator/offline_lane.py::OfflineLaneWorker` | h13, h9, comments, kind:cross-module | low | mechanical | confirmed | open |
| COORD-8 | fk-c9935b02f118 | orchestrator | `orchestrator/src/orchestrator/rebase_cost_readout.py` | h6, kind:dead-code | low | mechanical | weakened | open |
| SVG-16 | fk-132044d203a5 | orchestrator | `orchestrator/src/orchestrator/scheduler.py::Scheduler.set_task_status` | h12, h11, kind:deep-read | low | mechanical | confirmed | open |
| SHL-10 | fk-bb04ae68c0b1 | orchestrator | `orchestrator/src/orchestrator/usage_gate.py` | h13, tests, inv-5, kind:cross-module | low | mechanical | confirmed | open |
| COORD-10 | fk-f56fced3bfea | repo | `orchestrator/src/orchestrator/config.py::PRIORITY_TIERS` | h11, inv-5, h12, kind:cross-module | low | mechanical | confirmed | open |
| DASH-17 | fk-f5634bfcc909 | repo | `review/briefing.yaml` | inv-9, h11, kind:convention | low | mechanical | confirmed | open |
| DASH-16 | fk-3e77aa904a0a | sampler | `sampler/src/sampler/store.py::LoadSampleStore.cleanup_old` | comments, h13, inv-9, kind:deep-read | low | mechanical | confirmed | open |
| SHL-16 | fk-b83311fe5621 | shared | `scripts/legibility/coder.py` | h13, inv-5, kind:cross-module | low | mechanical | confirmed | open |
| COORD-4 | fk-1d7049ce6330 | shared | `shared/src/shared/api_health.py::ApiHealthGate` | h1, kind:dead-code | low | mechanical | confirmed | open |
| SHL-11 | fk-d6d7b2abfc5c | shared | `shared/src/shared/async_sqlite_base.py` | inv-5, h11, inv-11, h10, kind:cross-module | low | mechanical | confirmed | open |
| SHL-9 | fk-f7f1747302dc | shared | `shared/src/shared/usage_gate.py` | h14, h6, h13, h7, kind:deep-read | low | structural | confirmed | open |
