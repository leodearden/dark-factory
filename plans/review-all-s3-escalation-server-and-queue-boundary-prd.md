# S3 — Escalation server and queue boundary (PRD)

**Status:** draft, authored 2026-11-01 by the S3 author seat of the `/review-all` remediation program
(`plans/review-all-program-dark_factory-2026-10-10.md`, stream S3, spawn mode run as an agent seat,
size L, risk medium); critic findings folded 2026-11-02 by the S3 fold seat (two passes; lead rulings OQ-3, OQ-4, OQ-7(a) applied);
the sitting's rulings of 2026-11-10 applied the same day (§Rulings); fold-verified 2026-11-10, with the
lead's sequencing rulings of that day applied (§Rulings). B+H: contracts C1–C8
and a two-way boundary-test sketch. Twelve leaves, one of them the integration gate; the author's η
is task 5409's (B2). Implements ruling **R10** without relitigating it. Five leaves (ε1, ε2, ζ, λ, μ)
wait on the merge-lane lifecycle plan's native lane-query task (S6 μ) by the OQ-1 ruling. The lead's
`harness.py` order puts S6's first task α (the timing-budget validator, INV-15) before β, so β, γ, δ,
θ, ι and κ wait on S6 α (not on S6's native-port chain); α waits only on S6 α (S6 α lands first on every file it shares with another plan: `merge_lane/types.py`, `orchestrator/tests/test_merge_queue.py`, the ratchet baseline).

**Code anchors** verified against main `e4b333a601` (2026-11-10), 2,566 commits after the draft's
`d40ba048e9`. Every count below is labelled `measured at e4b333a601` unless it names another source;
the live-store and runs.db figures were read the same day. Main reached `77693cb399` while this fold
ran; of this plan's paths only `merge_lane/types.py` and `merge_lane/worker.py` moved (task 8524), and
the anchors used here (`ItemLifecycleState`, `MergeCancelVerdict.state`, `_REGISTRY_STATE_TO_WIRE`,
`snapshot()`'s 460 lines) are unchanged there. Re-confirmed at `38a200d213` (fold-verify, 2026-11-10):
the 9 commits since `77693cb399` touch none of this plan's paths, and every sidecar check was re-run
there. Main moves fast — cite by symbol; re-locate at implementation time.

## Goal

The escalation package serves the ladder — file, read, triage, claim, resolve, promote — and nothing
else. The orchestrator registers its own merge-lane and operator tools onto the FastMCP instance
escalation hands back, fed by typed ports instead of `getattr(harness, '_merge_worker')` and
`snapshot()` dicts; the merge tools read the lane through the merge-lane plan's native implementation
of the port, with no interim adapter (OQ-1). The queue has one locked read-modify-write primitive under
every mutator, one thread-delivery contract for its callbacks, and a per-task landing index that makes
"every escalation ever filed for this task" true even after the archive is pruned.

What a user observes when this lands:

- `python -c "import escalation.server"` loads a package that imports nothing from `orchestrator`, at
  module level, inside a function or under `TYPE_CHECKING`; `escalation/tests/test_layering.py` pins
  it. The fused-memory reconciliation escalation server (port 8103) lists 13 tools, none of which
  answers "no orchestrator attached".
- The dark-factory escalation server (port 8102) still lists 29 tools under their present names. The
  16 merge/ops tools are registered by `orchestrator.mcp.merge_tools` and `orchestrator.mcp.ops_tools`;
  `mcp__escalation__merge_request` answers as before, plus a `lane_unanswerable` field that names the
  lane query that could not answer when the dedupe gate fell back to its fail-safe.
- `mcp__escalation__merge_status` and `merge_cancel`'s refusal answer a `MergeState` member for every
  live request: the lane's one internal enum maps onto the public vocabulary through one total table,
  so an unrecognised state cannot reach a caller (OQ-8).
- `mcp__escalation__get_task_escalation_history(task_id=…)` for a task whose records were pruned by the
  30-day retention answers `count: 0, pruned_count: N, pruned: [ids…]`, never a bare `count: 0`. On the
  live store (4,076 archived records in 31 dated dirs, measured at e4b333a601) the default
  archive-inclusive `get_by_task` drops from 316–411 ms to the root glob (8.4–9.7 ms) plus reading
  that task's own indexed archive records.
- A pending record is always found by `get_by_task(status='pending')` and `has_open_l1`, whichever
  code version wrote it: the pending tier stays the root glob (B1).
- `mcp__escalation__resolve_issue` run from a FastMCP threadpool worker wakes the orchestrator's
  waiting workflow on the orchestrator's loop thread, because the queue delivers its resolve callback
  where the registrant asked, not on whichever thread happened to call `resolve()`.
- A second `merge_request` for a branch already in flight is answered by `dedupe_merge_request` and
  recorded as `merge_deduplicated`; `coalesce` names train formation only in merge-lane code and in
  every new event.

## Background

The run `review-all-dark_factory-20261008` found `escalation/src/escalation/server.py::create_server`
to be one closure holding two unrelated programs: the escalation ladder, which closes over `queue` and
two injected lookups, and the merge-lane/operator tools, which close over `harness`, `merge_queue`,
`event_store` and `orch_config` and reach back into `orchestrator` with function-local imports. The
queue re-implements its locked read-modify-write in every mutator, has no per-task index, and fires its
resolve callback on the calling thread without saying so. Rulings this PRD implements: R10 (the stream),
R23 (no weakened key of S3's — none of the 15 is in the R23 list), R24 (launch). Program conventions:
the filing stamps, the freshness clause and the held-leaf form in the program doc's Shared conventions.

**Re-measured at `e4b333a601`** (2026-11-10). Since `6551b75aba` eight modules were extracted from
`server.py` — `merge_status_answer.py`, `mismatch_streak.py`, `grant_delivery.py`,
`project_root_guard.py`, `store_identity.py`, `sha_citations.py`, `tool_description_block.py`,
`categories.py` — and task **6319** (done, merged `0935e63d88`) extracted the amendment-framing
machinery into `escalation/src/escalation/amendments.py`; since the draft's `d40ba048e9` task **8422**
(done) added a ladder tool and `escalation/src/escalation/policy_gate.py`, task **5324** (done) a
uuid-prefix guard, task **5516** (done) a recut gate in `resolve_issue`, and task **5224** (done,
merged at `e4b333a601`) moved the harness sentinel role prefixes into `escalation/models.py`. The
cost grew anyway. Counting rules: line counts are `wc -l` / AST `end_lineno - lineno + 1`; "prose" is
a non-blank line inside a statement-level string or carrying a comment token (`tokenize`); call-site
counts are `git grep -F '<name>('` over `*/src/*` and `scripts/*` excluding test directories, prose
lines included.

- `server.py` 5,973 lines; 3,156 of 5,470 non-blank lines (0.58) are docstring or comment.
  `create_server` spans 4,721 lines with 54 nested functions and 29 functions decorated with
  `@mcp.tool()` (AST; a `git grep -c '@mcp.tool()'` says 30 because it also matches a comment in the
  markup-middleware block). The 29 tool docstrings total 99,454 characters raw (resolve_issue 17,877;
  merge_request 13,263; promote_to_l2 12,686; escalate_blocker 6,952; merge_cancel 6,001).
- Partition by captured inputs (AST): 13 ladder tools close over `queue`, the two lookups,
  `dedupe_config` and `queue_kind` (derived from `store_identity`), plus four harness touches —
  `resolve_issue` reads `harness.pre_increment_routing_tier` and
  `harness.granted_files_delivery_blocker`, `get_pending_escalations` reads
  `harness.scheduler.get_statuses` and `harness.is_workflow_active` through `_annotate_pins_recovery`,
  `resolve_escalation_shas` reads `harness.git_ops`. The thirteenth, `claim_policy_class` (task 8422,
  `db75f3b1ce`), calls `escalation/src/escalation/policy_gate.py::perform_claim` and touches no harness.
  `create_server` also takes `prefix_resolver` (task 5324), which registers the uuid-prefix guard on
  both servers. 16 merge/ops tools close over `harness`, `merge_queue`, `event_store`, `orch_config`,
  `merge_inflight_registry`, the `_registry` and `_waiters` objects `merge_request` and `merge_cancel`
  share, and — through `merge_cancel`'s `_escalate_workflowless_cancel` — `_submit_or_dedupe`, which
  closes over `queue` and `cfg`. No ops tool closes over `queue`.
- 11 function-local `orchestrator.*` imports in `server.py` plus two under `TYPE_CHECKING`; four more
  runtime reverse imports in `escalation/src/escalation/git_authority.py` (2) and
  `escalation/src/escalation/merge_lane_resolution.py` (2). By owner: `claim_warm_worktree` and
  `release_warm_worktree` (δ); `_cancel_in_lane`, `merge_cancel`, `_write_merge_lane_metadata`
  (`LaneMetadataWrite`), `_get_merge_journal` and both `TYPE_CHECKING` imports (ε1); the registry
  construction in `create_server` and four in `merge_request` (ε2). `orchestrator.git_ops._run`
  (private) is still imported inside `release_warm_worktree`. `git_authority.py` now reads task
  metadata through `Scheduler.read_task` (task 8336, `750b9a5ea0`); its only non-test importers are
  `server.py` and `merge_lane_resolution.py`, plus prose pointers in `harness.py`,
  `merge_lane/landing_evidence.py` and `agents/landed_sha_searches.py`.
- `_get_merge_worker` (`getattr(harness, '_merge_worker')`) has 6 call sites (`merge_request` ×2,
  `get_merge_queue`, `_live_answer`, `_cancel_in_lane`, `merge_relane`), three of them reading
  `snapshot()` by string key. `_get_terminal_retention` has 3 write-side sites (`merge_request`,
  `_cancel_in_lane`, `merge_cancel`) and no read side; no production code assigns
  `harness._terminal_retention`. `_get_merge_journal` probes `harness._merge_store`. The harness
  builds its worker in `Harness._start_merge_worker`, after the server is composed, so every tool
  reads the worker per call.
- `orchestrator/src/orchestrator/merge_lane/intake.py` holds `coalesce_or_enqueue_merge_request`
  (111 lines) and the three snapshot-dict predicates `_inflight_entry_is_stale`,
  `_inflight_worktree_is_stale`, `_snapshot_verify_state` (186 lines: 38 + 106 + 42, every raise or
  shape failure returning "not stale", i.e. dedupe); `live_snapshot` appears on 25 of its lines.
  `server.py` is the only producer of `live_snapshot=`. `SpeculativeMergeWorker.snapshot` is 460 lines
  and runs `two_layer_invariants`, `speculation_accounting_violations` and `worktree_ledger_violations`
  inline. No test drives the predicates with a real `snapshot()`; the fakes are
  `_verify_snapshot`/`_queued_snapshot` in `orchestrator/tests/test_merge_queue_c3_submit_identity.py`.
- The lane already holds a closed enum: `merge_lane/types.py::ItemLifecycleState` (10 members:
  QUEUED, LANE_BUFFERED, MERGING, AWAITING_VERIFY, REDISPATCH_PARKED, DISPATCHING, VERIFYING,
  GATE_REVERIFY, FINALIZING, TERMINAL). `snapshot()` renders it as strings through
  `worker.py::_REGISTRY_STATE_TO_WIRE`, which renames two values on the way (LANE_BUFFERED →
  `'queued'`, REDISPATCH_PARKED → `'awaiting_host'`), because the escalation server could only read a
  dict. `server.py::create_server.<locals>._map_live_state` maps six strings and passes any other
  through as a raw string, deliberately (pinned by
  `escalation/tests/test_merge_state_wiring.py::TestMergeStatusTier1EmitsLiveStates::
  test_unrecognised_raw_still_passes_through`; task 6827 asks to map three fall-through states).
  `merge_lane/types.py::MergeCancelVerdict.state` is typed `str | None`, "the `snapshot()` wire state";
  `merge_cancel`'s refusal arm maps it with `_map_live_state`. `'passthrough'` is no longer produced
  (one hit under `orchestrator/src`, a docstring saying it was deleted; the `snapshot()` docstring
  still lists it).
- `escalation/src/escalation/queue.py` 3,476 lines, 1,870 of 3,105 non-blank (0.60) prose;
  `EscalationQueue` 2,581 lines. Eight mutators carry their own lock → load → status check → write:
  `stamp_triage`, `claim_policy_class` (task 8422), `amend`, `declare_pin`, `attach_dedupe_child`,
  `add_members_to_l2` (root-only, pending-only) and `patch_resolution_metadata`,
  `note_suppressed_refile` (`_locate_path`, terminal-only). Three of them — `stamp_triage`,
  `claim_policy_class`, `declare_pin` — share the load half, `EscalationQueue._load_root_pending`
  (`0a0d1fbb24`), and each still takes its own lock and writes. `resolve` and `park` are a
  `get()`-under-lock variant, `_write_late_resolution` writes a terminal record in place, and
  `submit_resolved` is a second creator beside `submit` (production callers:
  `escalation/src/escalation/note_to_file.py` and `server.py` ×2). The cascade's `resolved_by`
  encoding now has one writer and one reader, `escalation/models.py::cascade_resolved_by` and
  `cascade_parent_id` (`343e658060`). Outside `EscalationQueue`, three writers hold
  `escalation_id_lock` and write records: the relocation passes of `escalation/src/escalation/sweep.py`,
  `fused-memory/src/fused_memory/reconciliation/backlog_policy.py::BacklogPolicy._merge_onto_persisted`
  and `fused-memory/scripts/backfill_gate_backlog_summaries.py`. `set_resolve_callback` is a two-line
  setter; the callback fires inline from `resolve`, `park` (and once per cascaded member) and
  `submit_resolved`. `resolve_issue`, `stamp_triage`, `claim_policy_class`, `declare_pin` and the
  ladder reads are plain `def` (fastmcp 3.2.2 runs them through `anyio.to_thread.run_sync`), except
  `get_pending_escalations`, which is `async def`; `promote_to_l2`, `escalate_*`, `amend_escalation`
  are `async def`. Task **4567** (pending, high) is mid-flight: `task/4567` carries 8 commits of
  2026-11-07 that edit `get_by_task`'s cross-tier pre-scan and `_locate_path`'s archive-listing memo
  and move seven in-process archive-inclusive calls off the event loop.
- `get_by_task` against the live `data/escalations` (8,080 root dirents, 8 root records, 4,076
  archived records in 31 dated dirs), 5 runs each: `status='pending'` 8.4–9.7 ms, default 316–411 ms.
  `get_by_task(` on 103 source lines; `has_open_l1(` on 54 (42 of them `.has_open_l1(` calls), which
  now also filters by `dedupe_fingerprint` (task 3211). The pending read is a fail-closed veto:
  `orchestrator/src/orchestrator/scheduler.py::Scheduler._phase_redispatch_stranded_blocked` flips a
  blocked task to pending when it finds no pending record ("the esc-3163 lesson"). Two long-running
  processes write each project's queue: that project's orchestrator, which also hosts the escalation
  MCP server on its own event loop, and the single system-wide fused-memory service, which writes from
  about ten modules (`middleware/curator_escalator.py`, `ticket_janitor.py`,
  `dead_letter_escalator.py`, `scope_violation_escalator.py`, `reconciliation/backlog_policy.py` among
  them) into every project's queue. They restart on separate clocks — the orchestrators through the
  fleet redeploy sweep, fused-memory through `scripts/restart-fused-memory.sh` or its own watchdog
  staleness backstop (`scripts/orchestrator-watchdog.py`, `FM_WATCHED_PATHS` includes
  `escalation/src/`, an 8-hour minimum interval) — so for a while after a release one of them can still
  run code that predates a queue change. Short-lived command-line writers (`python -m escalation
  submit`, the sweep CLI, `archive.main`, maintenance scripts) load current code when they start.
- `read_escalation_for_scan` is 181 lines, 143 of them docstring, and still says "all seven reads
  routed" and "ALREADY GUARDED elsewhere … filed as a follow-up"; there are 13 production call sites
  (queue 4, sweep 3, watcher 2, shadow_ruling 1, scripts 3). `iter_all_escalation_paths` has 9 external
  call sites (dashboard 2, fused-memory 1, orchestrator 1, scripts 4, shadow_ruling 1). 11 production
  `from escalation.queue import …` lines name a scan helper, plus 4 test modules. The late-resolution
  policy (`_stamped_resolution_class`, `_is_late_resolution_worth_capturing`,
  `_build_late_resolution`, `_MAX_LATE_RESOLUTIONS`, `_MAX_LATE_RESOLUTION_CHARS`) is still in
  `queue.py`.
- `archive.prune_archive(queue_dir, DEFAULT_RETENTION_DAYS=30)` runs as pass 3 of
  `sweep.run_startup_sweep`, which `create_server` runs at every server start in both harnesses, and
  from `archive.main`. `get_task_escalation_history` still promises "ARCHIVE-INCLUSIVE BY
  CONSTRUCTION … was an escalation EVER filed".
- The watcher wire contract has five homes: `harness.py::_WATCHER_ESCALATION_HEADERS` (title-case
  names, `'0,1'`, the identity), `server.py::_LEVELS_HEADER`/`_IDENTITY_HEADER` (lowercase — required:
  fastmcp 3.2.2 `fastmcp/server/dependencies.py::get_http_headers` returns a dict keyed by
  `name.lower()`), `authority.py::_WATCHER_AUTO_IDENTITY` ("Must equal
  orchestrator.harness._WATCHER_ESCALATION_HEADERS['X-Escalation-Identity']"),
  `classify.py::_AUTO_WATCHER_RESOLVERS` and `scripts/sitting/apply.py::IDENTITY_HEADER`. Five test
  files import the harness constant (43 references). Five ladder tools read the headers:
  `resolve_issue`, `stamp_triage`, `claim_policy_class`, `amend_escalation`, `promote_to_l2`. The
  sentinel role prefixes the headers are meant to sit beside are already one constant,
  `escalation/models.py::HARNESS_SENTINEL_ROLE_PREFIXES` (task 5224).
- `merge_request`'s `workflow_verify` payload is `{passed, base_sha, branch}` under a "SYNC CONTRACT"
  comment; `workflow.py::TaskWorkflow._emit_workflow_verify` writes `passed, tip_sha, base_sha,
  branch, duration_ms, slot_wait_secs, site, runner` plus `verify_event_fields(result)` (`runner`
  added by task 6670, done `f6a4482199`). In `data/orchestrator/runs.db`, 78 of 2,441
  `workflow_verify` greens in the 30 days to 2026-11-10 carry no `tip_sha` key (363 since
  2026-07-19), all of the escalation-side shape. Task **5409** (pending) already carries this fix (B2;
  see §Findings disposition).
- "coalesce" in two senses. Every emitter of `EventType.merge_coalesced` uses the dedupe sense:
  `intake.py::_coalesce_onto_entry` (via `emit_merge_coalesced`),
  `intake_buffers.py::MergeIntake._coalesce_reentrant_drain` (`source='duplicate_submission'`) and
  `workflow.py::TaskWorkflow._submit_to_merge_queue` (a registry attach). The train sense is
  `train_coalesced`. No production code reads the `merge_coalesced` value; runs.db holds 20 rows, the
  last on 2026-09-18 (re-read 2026-11-10). fused-memory's
  `reconciliation/coalescing.py::merge_coalesced_payload` is reconciliation-event folding, a third
  domain outside the merge lane, and is not renamed.
- Tests coupled to `create_server`'s present signature: 207 `create_server(` calls in 25 test files
  pass at least one keyword ζ deletes (`harness=`, `merge_queue=`, `orch_config=`, `event_store=`,
  `merge_inflight_registry=`, `task_status_lookup=`, `task_claimant_lookup=`), 395 such keyword
  arguments (AST). `scripts/tests/test_sitting_payloads.py` AST-parses `server.py` for `resolve_issue`
  and `RESOLVE_ACTIONS`; `scripts/tests/test_merge_state_vocabulary_consistency.py::PINNED_SITES`
  keys `merge_cancel`'s span on the `server.py` path; `escalation/tests/test_git_authority.py` and
  `orchestrator/tests/test_citation_gate_positive_arm.py` import `escalation.git_authority`; the
  merge-lane ratchet baseline `orchestrator/tests/merge_lane_ratchet_baseline.json` keys
  `intake.py::coalesce_or_enqueue_merge_request`. `fused-memory/tests/test_stages.py::
  test_every_escalation_server_tool_is_classified` enumerates the tools `create_server` registers and
  `fused-memory/tests/_tool_surface_classification.py::assert_tool_surface_classified` fails on a
  bucket that names a tool the server does not register: 15 of the 16 merge/ops tools are listed in
  `test_stages.py::_REVIEWED_STAGE_SAFE`, and `merge_relane` is the whole of
  `fused-memory/src/fused_memory/reconciliation/cli_stage_runner.py::DISALLOW_ESCALATION_MERGE_LANE_CONTROL`,
  so every leaf that moves a tool off `create_server` edits those lists. 81 tracked files outside
  `plans/` and `review/` (JSON excluded) hold a `server.py::` or `escalation.server.` prose pointer (81
  at `d40ba048e9` by the same rule).

## Findings disposition

Every key of the stream appears once. Step 1 of the contract §8 dedup (`find_tasks_by_metadata` on
`x_finding_key`) returns no task for any of the 15 keys (re-run for all 15 on 2026-11-10: empty).
Step 2 (`search_tasks` on each leaf's remedy, ≥ 0.6 confirmed with `get_task`, re-run 2026-11-10)
finds task 5409 carrying `fk-bc64099a1cd0`'s remedy and task 4833 carrying `fk-66746efddd4d`'s (its
description deletes `server.py::_get_terminal_retention` and every `retention=` thread; by the lead's
ruling of 2026-11-10 it lands first), and no new live task for any other key; the
other hits are done tasks this PRD builds on or the live tasks in §Live tasks. No finding was fixed
on main between `d40ba048e9` and `e4b333a601`; two moved partway (noted per row).

| Key | Sev | Disposition | Re-verified at `e4b333a601` |
|---|---|---|---|
| `fk-a8d3003e9d83` | high | leaf δ, ε1, ε2, ζ (gate μ) | create_server 4,721 lines, 54 nested functions, 29 tools, two programs in one closure |
| `fk-e903cc072b15` | medium | leaf δ (2 imports), ε1, ε2 (merge side), ζ (zero-import guard) | 11 + 2 reverse imports in server.py, 4 in git_authority/merge_lane_resolution; private `_run` still imported |
| `fk-43f431f697a3` | medium | leaf λ | 5,973 lines, 0.58 prose, 99,454 chars of tool docstrings over 29 tools |
| `fk-609a1832903b` | medium | leaf ι; the `_chokepoint_or_submit` third is S1's (R8 `filing.py`) | resolve_issue 576 lines / 272-line docstring, two more gates since (5516's recut conflict, 8422's identity policy refusal); promote_to_l2 493 / 218; `_chokepoint_or_submit` 184, in-place mutation |
| `fk-4a0626fc033d` | high | leaf β; the "resolve_issue's unlocked `_rewrite`" clause is `fixed:49207cc422` (task 6617) | 8 copies (task 8422 added `claim_policy_class`); 3 of them share the load half `_load_root_pending` (`0a0d1fbb24`) but still lock and write themselves; plus the `resolve`/`park` variant and `_write_late_resolution` |
| `fk-5a9e713f4055` | medium | leaf γ; the amendment-framing half is `fixed:0935e63d88` (task 6319) | 0.60 prose; scan docstring 143 lines and stale ("all seven" vs 13 sites); late-resolution policy still in queue.py |
| `fk-e5d81cc1f1d4` | medium | leaf θ | 8.4–9.7 ms pending / 316–411 ms default on 4,076 archived records |
| `fk-07be43f313bc` | high | leaf β | setter unchanged; thread depends on `def` vs `async def` spelling |
| `fk-f8e9e0da3786` | medium | leaf ε1, ε2 (port in α; native implementation S6 μ) | `_get_merge_worker` 6 sites; `harness` typed `Any` |
| `fk-66746efddd4d` | medium | `filed:4833` (lead action: amend 4833 with the key and today's paths; 4833 lands FIRST, in today's `server.py` and `merge_lane/`, deleting `_get_terminal_retention`, its 3 `retention=` sites and the ring type; ε1 depends on it — lead ruling 2026-11-10) | read side gone (`234c5e1d4f`); 3 write sides in `server.py` pass a ring nobody constructs (`retention=` also threads `intake.py`, `gates.py`, `worker.py`, `queue_store.py`; measured at 38a200d213) |
| `fk-bc64099a1cd0` | medium | `filed:5409` (lead action: amend 5409 with the key, the constructor contract and `runner`; 5409 lands FIRST, in today's `server.py`, and ε2 depends on it — OQ-1 ruling) | payload still `{passed, base_sha, branch}`; 78 of 2,441 greens in 30 days carry no `tip_sha` key, all from this emitter |
| `fk-ce2dc99e0305` | medium | leaf θ | prune at every server start, 30 days; promise text unchanged |
| `fk-43bad4e724da` | low | leaf κ | five homes (`scripts/sitting/apply.py` added one), five tools read the headers, five test files import the harness constant; the sentinel prefixes already moved to models.py (task 5224) |
| `fk-c8c3406fd7f4` | medium | leaf α (port) and ε2 (gate onto the native port; the three predicates deleted) | predicates in intake.py (186 lines); snapshot() 460 lines with three inline audits; fakes only in tests |
| `fk-0de72a91f89a` | low | leaf α | every `merge_coalesced` emitter is dedupe-sense; six more dedupe-sense identifiers |

## Pre-conditions for activating

- **Task 6319** is done (`0935e63d88`): `amendments.py` exists with public `MAX_AMENDMENT_*`, `elide`,
  `append_amendment_capped`, `has_framing`, `amendment_room`. No leaf depends on it.
- **Task 4833** (pending, low; `plans/merge-status-durable-non-landed-prd.md` ε; dependencies 4830,
  4831, 5036 all done, so it is dispatchable today) deletes `TerminalOutcomeRetention`,
  `server.py::_get_terminal_retention` with its three `retention=` sites (`merge_request`,
  `_cancel_in_lane`, `merge_cancel`) and the lane-side `retention=` threads (`merge_lane/types.py`,
  `intake.py`, `gates.py`, `worker.py`, `queue_store.py`; measured at 38a200d213). Its description still
  names `merge_queue.py`/`merge_types.py`, which no longer exist (lead amend). By the lead's ruling of
  2026-11-10 the edge is REVERSED, like 5409's and 4651's: 4833 lands FIRST, in today's files, and ε1
  depends on it, so the merge-tool moves carry code that no longer passes a ring. 4833 therefore carries
  `fk-66746efddd4d` (§Findings disposition) and waits on no S3 or S6 task (S6's limits task κ depends
  on 4833, not the reverse).
- **Task 5409** (pending, medium) owns the `workflow_verify` constructor (B2). By the OQ-1 ruling it
  lands FIRST, in today's files: it edits `server.py::create_server.<locals>.merge_request` where it is
  now, and ε2 depends on it and carries its call into `orchestrator/src/orchestrator/mcp/merge_tools.py`.
- **Task 4651** (pending, high; waits on 4646) adds the periodic landed reconciler, which imports
  `escalation.git_authority`. By the OQ-1 ruling it lands FIRST, against today's path; ε1 depends on it.
  By S8-2 = A it is re-scoped onto S8's `StrandedReconciler` and depends on S8 ε2a, so its
  `git_authority` import lives in `orchestrator/src/orchestrator/stranded_reconcile.py` (lead ruling
  2026-11-10); ε1 repoints that import there when it moves the module.
- **Task 4567** (pending, high, mid-flight on `task/4567`) edits `queue.py`'s `get_by_task` and
  `_locate_path`; θ depends on it, so the index starts from its result. It is a prerequisite of θ
  only, not of β (lead ruling 2026-11-10): β's mutator edits are region-disjoint from 4567's.
- **S6 α** (S6's timing-budget validator, INV-15) lands first on `harness.py` (lead ruling
  2026-11-10: S6 α is never held behind S3); β, S3's first `harness.py` leaf, depends on it, and S6's
  next `harness.py` task (β1) depends on κ, S3's last unheld one.
- **S6** builds the native `MergeLaneQueries` (S6 C5, its native-port task μ) in
  `orchestrator/src/orchestrator/merge_lane/queries.py`, which α creates, together with the total
  table `LIVE_STATE_OF` from the lane's `ItemLifecycleState` to `MergeState` (OQ-8). There is no
  interim adapter (OQ-1). ε1 and ε2 depend on S6 μ; S6 μ depends on S3 α and must NOT depend on any
  S3 merge-tool move (that edge would close a cycle). S6 μ keeps the priority S6 gives it; the
  scheduler's priority inheritance (`orchestrator/src/orchestrator/scheduler.py::
  Scheduler._compute_effective_priorities`: a task's effective priority is the highest of its own and
  its undone dependents') runs it, and S6's chain beneath it, at ε1/ε2's high priority once they are
  filed. Nothing else in S3 is held on S6 μ.
- **S7** deletes reify retry narrowing end to end (S7-3), including `merge_request`'s
  `retry_failed_only` parameter in today's `server.py`. That deletion (S7 δ, first in S7's chain by
  the lead's ruling of 2026-11-10) lands before ε2, so the move carries one parameter fewer. S6 α
  takes no edge on S7 δ or S7 β (nor on S5 α or S5 λ0); those tasks follow S6 α where they share a
  file, so β, δ, κ, θ and ι wait on S6 α alone and on no S5 or S7 task. On `server.py` S7 δ is
  region-disjoint from α, δ, κ, θ and ι and the lead's per-file order places it; S7 δ must not depend
  on ε2 or any later S3 leaf (ε2 → S7 δ, so that edge closes a cycle).
- **S8** owns `SlotRegistry`; it implements S3's `WorkflowSlots` (C2) and drops its own definition
  (OQ-7, ruled (a) by the lead 2026-11-02). Until S8 lands, `Harness` implements `WorkflowSlots`;
  `cancel` is `Harness.cancel_workflow`, which already stamps `scheduler.note_workflow_cancelled`.
  The S8 ↔ S3 cycle cut is confirmed by the lead (2026-11-10): S8 β depends only on S3 β and δ; S3 ε1
  depends on S8 ε2b; S8 ε3 and S8 ω (which deletes the `is_workflow_active` forwarder) depend on S3 ζ.
- `shared/src/shared/git_async.py::run_git` exists today (S5 owns it). S5's C1 table assigns the
  `release_warm_worktree` repoint to S3 δ, and S5 β (which deletes `orchestrator.git_ops._run`)
  depends on δ; nothing in S3 waits on S5.
- fastmcp 3.2.2 (re-checked in the installed package 2026-11-10): `FastMCP.tool(fn)` and
  `add_tool(fn)` register after construction; `add_transform(AppendToEveryToolDescription)` and
  `add_middleware(...)` cover tools registered later; `FastMCP.instructions` has a setter;
  `get_http_headers()` lower-cases header names.

## Sketch of approach

Order of work (OQ-1 ruling and the lead's sequencing rulings of 2026-11-10). Seven leaves do not wait
on S6's native-port chain: α, β, γ, δ, θ, ι, κ (β, and through it γ, δ, θ, ι, κ, wait only on S6's
first task α, which lands first on `harness.py`). Five wait on S6's native lane-query task (S6 μ): the
two merge-tool moves ε1 and ε2, which read the lane through it, and — through the serial chain on
`server.py` — the ladder split ζ, the prose pass λ and the gate μ. The header constants κ, the history
index θ and the gate tables ι are therefore placed before the ladder split and edit the tools where
they live today; ζ then moves their code into the family modules, so that code moves twice. That is
the price of not holding three live fixes. κ goes first of the three on `server.py`: S6's `harness.py`
chain (β1 … μ) follows κ, so placing κ ahead of θ keeps task 4567 (θ's prerequisite) and θ, ι off
S6's path and therefore off ε1/ε2's.

**α — the lane's typed query port and the dedupe rename.** `orchestrator/merge_lane/queries.py` (new)
defines C1: `MergeLaneQueries`, `LiveRequestState`, `TerminalRequestState`, `QueryUnavailable`. There
is no adapter over `snapshot()` (OQ-1): α ships the Protocol, the two state records and the error, and
the merge-lane plan's native-port task (S6 μ) adds the implementation and the `LIVE_STATE_OF` table to
this file. `LiveRequestState.state` is a `MergeState` and never empty (OQ-8): the lane's internal
`ItemLifecycleState` reaches it only through `LIVE_STATE_OF`, a total table whose exhaustiveness a test
checks, so no raw string crosses the port. `intake.py`'s three snapshot-dict predicates stay until ε2:
nothing typed can answer the dedupe gate before S6 μ. The closed rename set (OQ-2, ruled (a)):
`coalesce_or_enqueue_merge_request` → `dedupe_merge_request`, `emit_merge_coalesced` →
`emit_merge_deduplicated`, `InFlightMergeRegistry.record_coalesced` → `record_deduplicated`,
`_coalesce_onto_entry` → `_dedupe_onto_entry`, `MergeIntake._coalesce_reentrant_drain` →
`_dedupe_reentrant_drain`, `InFlightEntry.coalesced_request_ids` → `deduplicated_request_ids`,
`_alarm_coalesce_live_original` → `_alarm_dedupe_live_original`,
`_COALESCE_LIVE_ORIGINAL_SENTINEL_PREFIX` → `_DEDUPE_LIVE_ORIGINAL_SENTINEL_PREFIX` (the constant
only). All three emitters write a new `EventType.merge_deduplicated` and keep their `source`
discriminator; `EventType.merge_coalesced` stays as a read-only legacy member for the 20 historical
rows. Two stored-vocabulary exceptions keep their values: the escalation category
`merge_coalesce_live_original` (`escalation/src/escalation/categories.py`, S2's), and the sentinel
task_id prefix `'__merge_coalesce_live_original__'` — `intake_buffers.py::_alarm_coalesce_live_original`
files under `task_id=<prefix><request_id>` and dedupes on `escalation_queue.has_open_l1(sentinel)`, so a
new value would hide old-prefix open L1s and refile (R8's dunder sentinel literals: S1 policy, S2
registry). `server.py`'s `merge_request` follows the rename at its import and call. Comment mentions of
the old gate name in `worker.py`, `queue_store.py` and `tip_relation.py` are S6's to retarget (xref).
The ratchet baseline's `intake.py::coalesce_or_enqueue_merge_request` key is regenerated with
`--authorize-raise`, as task 5305 did. Train-sense names are untouched.

**β — one mutation primitive and one callback contract in the queue.** `_mutate_pending(id, mutate,
*, archive_after=False)` and `_mutate_terminal(id, mutate)` (C4, D12) are the only `EscalationQueue`
methods that write an existing record. Each mutator becomes a `mutate(esc) -> bool` plus a mapping
from `MutationResult` to its public return shape; the eight copies (task 8422's `claim_policy_class`
included), the `resolve`/`park` `get()`-under-lock variant and `_write_late_resolution`'s in-place write
go onto the primitives, and `_load_root_pending` (today shared by three of them) becomes the
primitive's load step. `submit_resolved` stays a creator: it and `submit` write through one shared
lock-and-write helper. `set_resolve_callback(cb, *, deliver_on=None)` and `set_notify_callback` state
the contract: with a loop, delivery is `call_soon_threadsafe` from any other thread and inline on that
loop; without one, inline on the caller's thread, which the docstring now says. Called off the loop
with `deliver_on`, `resolve()` returns before the callback has run (a stated timing change; no caller
is known to depend on the old order, and the leaf audits the `resolve()` callers it can find).
`call_soon_threadsafe` on a closed loop raises `RuntimeError`: the queue catches it, logs a WARNING
naming the record and the callback, counts the drop, and never raises into `resolve()` (INV-11). The
harness registers with its loop. `event.set()` on an `asyncio.Event` stops running on a threadpool
thread. β follows S6 α on `harness.py` (lead order); task 4567 is θ's prerequisite, not β's.

**γ — the queue's pure helpers in leaf modules.** `escalation/scan.py` (new) takes
`iter_all_escalation_paths` and `read_escalation_for_scan` with a short contract docstring; the
143-line inventory docstring is deleted (the call-site count lives in this PRD). The late-resolution
policy (`_is_late_resolution_worth_capturing`, `_build_late_resolution`, `_stamped_resolution_class`,
`_MAX_LATE_RESOLUTION*`) joins `amendments.py` as public names. Every importer is repointed — 11
production import lines and 4 test modules — with no re-export shim in `queue.py`; prose pointers to the
moved names (`models.py`'s pointers to `queue._build_late_resolution`/`queue._MAX_LATE_RESOLUTIONS`,
`dashboard/src/dashboard/data/escalations.py`, `scripts/escalation_ladder.py`,
`scripts/tests/test_escalation_ladder.py`, `escalation/tests/_scan_race_helpers.py`) are retargeted.
`make_id`, `_recover_seq_from_disk`, `add_members_to_l2`, `submit`, `get`, `_locate_path` docstrings
are cut to the contract under the Comments stance.

**δ — operator tools registered from the orchestrator.** `orchestrator/mcp/ops_tools.py` (new):
`register_ops_tools(mcp, *, ops: HarnessOps, slots: WorkflowSlots, scheduler, git_ops)` registers
`release_workflow`, `halt_merge_queue`, `unhalt_merge_queue`, `get_merge_halt_status`,
`halt_scheduler`, `resume_scheduler`, `reload_config`, `clear_verify_host_quarantine`,
`get_task_runtime_state`, `claim_warm_worktree`, `release_warm_worktree`. `HarnessOps` (C2) is the
Protocol of the public harness methods they call; `WorkflowSlots` (C2) is the slot read surface S8's
`SlotRegistry` will implement (OQ-7, ruled (a)); `Harness` implements it in the interim,
`wait_cleared` polling `is_workflow_active` as `release_workflow` does today, and `release_workflow`'s
poll loop becomes `await slots.wait_cleared(...)`. `WorkflowSlots.cancel` includes the R3 grace stamp
(`scheduler.note_workflow_cancelled(task_id)` when it returns True), so binding S8's registry must keep
it. δ repoints `release_warm_worktree` onto `shared.git_async.run_git` while moving it (S5 C1's table
assigns that edit to δ). The eleven names leave `fused-memory/tests/test_stages.py::_REVIEWED_STAGE_SAFE`
in the same commit: the recon stage classification test fails on a bucket naming a tool `create_server`
no longer registers, and the reconciliation server never serves these tools. Tactical in δ: whether
`release_workflow` reports the slot's `slot_kind` when `cancel` returns False for a running
deterministic slot (S8's OQ-5). `orchestrator/tests/_escalation_server_factory.py::dark_factory_server(queue,
*, lane_view, ops, slots, …)` (new) is the one test builder of the composed server, typed against
C1/C2/C3, so each migrating test call site changes by one substitution. Their tests move to
`orchestrator/tests/`. The halt/unhalt tools keep today's answers; S6's C7 `HaltState` owner semantics
reach them through S6's xref, with no C2 change now.

**θ — the per-task landing index for the archive tier; honest history.** C5:
`data/escalations/by-task/<encoded task_id>.jsonl`, one line per record filed, appended by `submit`
and `submit_resolved` after the record is durable, under the record's lock. The pending tier keeps
today's root glob (OQ-6, ruled (a)): the root is the ground truth for open records, so a record written
by a process that predates θ is still returned by `get_by_task(status='pending')` and `has_open_l1`
(B1). The index drives only the archive tier: the default read is the root glob ∪ the index-located
archive records, deduplicated by id, each read through `get()`, which keeps its move-race retry (task
5118's race is retried, not gone). Pass 0 of `run_startup_sweep` backfills the index from a full scan
(idempotent by id). `archive.prune_archive` and `archive.main` append a tombstone line for each id they
delete, so "pruned" is a positive fact. `get_task_history(task_id)` returns live records and the
tombstoned entries; `get_task_escalations` and `get_task_escalation_history` add `pruned_count` and
`pruned` to their envelopes in `server.py`, where they live until ζ moves them, and their
evidence-of-absence text becomes true. The id-key/task-id divergence (task 4451) is a non-issue: the
index is keyed by the record's `task_id`. Rollout (INV-13's backfill decision, per record kind):
pending → found by the root glob; archived before θ → indexed by pass 0; archived after θ by a writer
still running older code — the project's orchestrator or the fused-memory service, which restart on
separate clocks (§Background) — → indexed by that queue's next pass 0, which is why μ's deploy signal
names a fused-memory restart too. θ starts from task 4567's result (`get_by_task`'s pre-scan and
`_locate_path`'s memo, mid-flight on `task/4567`).

**ι — gate tables for `resolve_issue` and `promote_to_l2`.** `escalation/ladder_gates.py` (new; C8):
`resolve_issue`'s steps — the fk-609a1832903b list of 12 plus the two gates added since,
`escalation/src/escalation/models.py::recut_action_conflict` (task 5516) and `escalation/src/escalation/policy_gate.py::identity_resolve_refusal`
(task 8422) — are classified once as parse, gate, branch or effect, and `promote_to_l2`'s steps are
classified separately the same way. Parse steps (terminate migration, `resolution_class` derivation)
build a frozen `GateInput`; the park branch is selected before the gates run; gates return
`Refusal | None`, and the ceiling and its auto-close carve-out are one gate (the carve-out is consulted
only when the ceiling would deny, as `escalation/src/escalation/authority.py::l2_auto_close_class`
states); effects (pre-stamp, routing-tier bump, late-resolution payload) run after the last gate.
`RESOLVE_GATES` and `PROMOTE_GATES` are ordered tuples; `first_refusal(gates, ctx)`. A new gate is one
function and one tuple entry. ι edits the two tools in `server.py`; ζ moves them with their tables
wired. `_chokepoint_or_submit`'s five gates are S1's (R8) and are not touched here.

**κ — the watcher wire contract has one home.** C6 in `escalation/models.py`, beside
`HARNESS_SENTINEL_ROLE_PREFIXES`: `LEVELS_HEADER`, `IDENTITY_HEADER`, `WATCHER_AUTO_IDENTITY`, and the
one reader `read_caller_headers(headers) -> CallerHeaders(levels_raw, identity)`, which looks up
`LEVELS_HEADER.lower()` and `IDENTITY_HEADER.lower()` (fastmcp lower-cases names). No other code reads
these headers; the five tools that read them today (`resolve_issue`, `stamp_triage`,
`claim_policy_class`, `amend_escalation`, `promote_to_l2`, still in `server.py`) call it.
`authority.py::watcher_connection_headers()` derives the levels value from
`ROLE_LEVEL_ALLOWLIST[WATCHER_AUTO_IDENTITY]`, so `'0,1'` is spelled nowhere. The harness,
`classify.py`, `scripts/sitting/apply.py` and the five test files import. S2's `parse_caller_identity`
absorbs `read_caller_headers` later.

**ε1 — merge read/cancel/relane tools registered from the orchestrator (waits on S6 μ).**
`orchestrator/mcp/merge_tools.py` (new): `register_merge_tools(...)` (C7a) registers `merge_status`,
`merge_cancel`, `merge_relane`, `get_merge_queue` with the tier walk (`_live_answer` … `_resolve`),
`_cancel_in_lane`, `_escalate_workflowless_cancel`, `_write_merge_lane_metadata`, `_PollKey`,
`_DurableHistory`, `_ProbeFailed`, `_project_task_runtime_entry`. The harness wires S6's native
`MergeLaneQueries` implementation as the `lane_view` provider (the worker is built after the server is
composed, so the tools read it per call). `_live_answer` reads `lane_view().live_states_for(...)` and
answers the record's `MergeState`; `merge_status` reads the lane's `terminal_state` tier from the start,
because the native implementation answers it from the journal and `merge_finalized` rows (INV-13).
`get_merge_queue` keeps `worker.snapshot()` because it is the rendering consumer; that is a display,
not a contract, and its stale `'passthrough'` docstring entry goes. `merge_cancel`'s refusal arm reads
a typed verdict: ε1 retypes `merge_lane/types.py::MergeCancelVerdict.state` from the wire string to
`ItemLifecycleState | None` (and its one producer in `worker.py`, which stops going through
`_REGISTRY_STATE_TO_WIRE`) and maps it through `LIVE_STATE_OF` (OQ-8, ruled). `_get_merge_journal`,
`_map_live_state` and their `hasattr(worker, 'snapshot')` guards are deleted, with no copy of the
mapping in `merge_tools.py`; the journal is the explicit `journal` argument. The pinned
`test_unrecognised_raw_still_passes_through` is deleted with `_map_live_state`; the exhaustiveness test
on `LIVE_STATE_OF` (S6 μ's) replaces it, and ε1's commit message says it deliberately reverses the
forward-compatible pass-through. Moves (git mv; their reverse imports become ordinary downward
imports): `escalation/git_authority.py` → `orchestrator/mcp/git_authority.py` (OQ-4, ruled by the lead
2026-11-02), `escalation/merge_lane_resolution.py` → `orchestrator/merge_lane/lane_resolution.py`,
`escalation/merge_status_answer.py` → `orchestrator/mcp/merge_status_answer.py`; the import task 4651
adds against `escalation.git_authority` — inside `orchestrator/src/orchestrator/stranded_reconcile.py`,
where 4651 lands on S8's `StrandedReconciler` (S8-2) — is repointed in the same commit. Task 4833 has
already deleted `_get_terminal_retention` and the `retention=` keywords, so the moved tools pass no ring. The ladder's filing path the
moved `merge_cancel` needs is exported, not reached into (M5): ε1 lifts the `_submit_or_dedupe` closure
into the public `escalation/filing_tools.py::submit_or_dedupe(queue, esc, *, dedupe_config)` (a thin
wrapper over `escalation/src/escalation/dedupe.py::submit_or_dedupe_off_loop` that keeps the
born-at-L2 bypass), and the composition root binds it as `file_escalation`.
`_escalate_workflowless_cancel` mints its record id through `mint_escalation_id` (bound to
`queue.make_id`, called via `asyncio.to_thread`, as today). `merge_request` and `merge_cancel` share
the in-flight registry and the waiter store: from ε1 the harness constructs both
(`InFlightMergeRegistry`, and `MergeWaiters`, which lives in `orchestrator/mcp/merge_tools.py`) and
hands the same objects to `register_merge_tools` and, until ε2, to `create_server` through its existing
`merge_inflight_registry=` and an interim `merge_waiters=` keyword typed `Any` (no `TYPE_CHECKING`
orchestrator import comes back). `merge_relane` leaves
`fused-memory/src/fused_memory/reconciliation/cli_stage_runner.py::DISALLOW_ESCALATION_MERGE_LANE_CONTROL`
(the reconciliation server no longer serves it) and the three other names leave
`_REVIEWED_STAGE_SAFE`. Test call sites migrate through `dark_factory_server`.

**ε2 — `merge_request` registered from the orchestrator; the gate on the native port; the probes go
(waits on S6 μ).** `merge_request` and its `dedupe_merge_request` call move into `merge_tools.py`,
carrying task 5409's constructor call and one parameter fewer than today (S7's deletion of
`retry_failed_only` lands first). The dedupe gate moves onto the port: `_c3_submit_gate`, the
corpse-reaper and the stale-slot reaper read `lane.live_request_state(...)`,
`lane.live_states_for(branch=...)` and `lane.owns_merge_worktree(...)`; `intake.py`'s three
snapshot-dict predicates (186 lines) and the `live_snapshot` plumbing are deleted. An in-verify entry
(`verify_started_at` set) goes to
`orchestrator/src/orchestrator/merge_lane/tip_relation.py::decide_c3_submit_action`: SUPERSET → REJECT
`duplicate_in_verify`; SAME/SUBSET → dedupe (task 2927's D1: same branch + same SHA never rejects). A
`QueryUnavailable` keeps today's documented fail-safe (dedupe), logs it, emits one event, and sets
`MergeDispatchResult.lane_unanswerable` to the name of the query that raised; `merge_request` returns
that field, so the fail-safe is visible to the caller (INV-11). `_get_merge_worker`, the lazy
registry construction, the `merge_inflight_registry=` and
`merge_waiters=` keywords and the function-local imports in `merge_request` are deleted; after ε2
`escalation/src` holds no `from orchestrator` line. `merge_request` leaves `_REVIEWED_STAGE_SAFE`.
`review/briefing.yaml`'s "duck-typed harness._merge_worker" known gap is removed (contract §7: the
closing task is this leaf). `server_instructions.py` loses its merge/ops sentence;
`register_merge_tools` appends `MERGE_TOOLS_INSTRUCTIONS` through the `instructions` setter.

**ζ — the ladder registered from escalation modules; `create_server` becomes composition.** Three
family modules (OQ-3, ruled by the lead 2026-11-02): `escalation/filing_tools.py` (adds
`escalate_info`, `escalate_blocker`, `_chokepoint_or_submit`, the markup filers and storm reporters
beside ε1's `submit_or_dedupe`), `escalation/resolve_tools.py` (`resolve_issue`, `promote_to_l2`,
`stamp_triage`, `claim_policy_class`, `declare_pin`, `amend_escalation`, the L2 derivation helpers,
`RESOLVE_ACTIONS`/`_DISMISS_ACTIONS`, with ι's gate tables and κ's header reader already wired) and
`escalation/read_tools.py` (`get_pending_escalations`, `get_task_escalations`,
`get_task_escalation_history` with θ's envelopes, `get_escalation`, `resolve_escalation_shas`,
`_annotate_pins_recovery`, `_compact_escalation`). Each exposes `register_*(mcp, *, queue, lookups,
guard, ...)`. `escalation/lookups.py` (new) defines `LadderLookups` (C3), the only thing the ladder
ever learns about an orchestrator; `harness: Any` leaves the package. `create_server(queue, *,
dedupe_config, lookups, startup_sweep, startup_sweep_now, store_identity, prefix_resolver,
mismatch_streak_threshold)` builds the `FastMCP`, middleware, transform, guards and sweep, calls the
three registrars and returns. `escalation/tests/_ladder_server.py::ladder_server(queue, *, lookups=…)`
(new) is the ladder test builder; every ladder test call site migrates by one substitution.
`escalation/tests/test_layering.py` pins "no `orchestrator` import anywhere under `escalation/src`"
(module-level, function-local and `TYPE_CHECKING`, by AST) and "`server.py` registers no tool itself".
`scripts/tests/test_sitting_payloads.py` reads each tool's input schema from `FastMCP.list_tools()`
instead of parsing `server.py` source (Tests stance). The fused-memory harness's
`create_server(queue, store_identity=..., prefix_resolver=...)` call is unchanged.

**λ — prose under the Comments stance.** Every tool description states the agent-facing contract and
nothing addressed to a maintainer or reviewer: no task numbers, no incident narratives, no "please do
not fix it". Durable rationale moves to the commit message or the module that owns the decision, with a
one-line pointer; which home per tool is tactical. Preservation criterion: the leaf lists every tool's
agent-directed imperatives (MUST, NEVER, "treat … as") before and after and diffs the two lists in its
commit; an imperative that disappears is a defect. Module docstrings of the new/moved modules state
purpose in one sentence.

**μ — integration gate and companion docs.** `_serve_escalation_mcp_impl` moves from
`escalation/tests/conftest.py` into `escalation/src/escalation/testing_http.py` (importable; takes a
server factory, imports nothing from `orchestrator`), which both conftests use. The boundary suite
runs over real HTTP in two halves: the escalation suite (ladder-only shape, rows 4–8, 10–14, 16, 17)
and the orchestrator suite (both registrars, rows 1–3, 9, 15). `ARCHITECTURE.md` and
`skills/merge-queue/references/two-layer-model.md` name `dedupe_merge_request` and the two registrars.

## Resolved design decisions

- **D1 — the port is orchestrator-internal (R10 option 1).** `MergeLaneQueries` lives in
  `orchestrator/merge_lane/queries.py`, imported downward by `intake.py`, `mcp/merge_tools.py` and (S6)
  `worker.py`. Escalation never sees it. The G4 row "S3 defines, S6 implements, escalation server
  consumes" is read as: the *served* escalation server consumes it through the orchestrator-registered
  tools. S6's C5 implements this Protocol at this import path (S6 μ, after S3 α; ε1 and ε2 wait on
  it). Rejected: a Protocol in `escalation/` or `shared/` — it would make escalation describe a lane it
  must not know.
- **D2 — no interim adapter; the merge tools wait for the native port (OQ-1, ruled (a) by Leo).**
  The five dict readers (three predicates, `_live_answer`, `merge_request`'s depth read) go straight
  onto S6's native implementation, which answers from the lane's own registry and journal without the
  audits inside `snapshot()`. No `SnapshotLaneQueries` exists at any point, so nothing deletes one.
  The cost is the hold: ε1 and ε2 (and, through the `server.py` chain, ζ, λ, μ) wait on S6 μ, which
  sits at the end of S6's serial chain behind task 5040 (S6-2 ruled (b)). The three predicates stay in
  `intake.py` until ε2, and the history index, gate tables and header constants are placed ahead of
  the ladder split so they are not held (their code moves twice). Rejected by the ruling: an interim
  adapter (one more class to build and delete) and a native implementation written by S3 (a parallel
  edit of S6's registry, G4).
- **D3 — queries raise, never guess; the fail-safe is visible.** `QueryUnavailable` replaces the seven
  "return False / (False, None) → dedupe" arms. The gate keeps the fail-safe *decision*, records it
  (log + event), and returns it to the caller in `MergeDispatchResult.lane_unanswerable`, which
  `merge_request` surfaces — INV-11 ("a log is not a return value"). ε2 builds it with the gate
  conversion.
- **D4 — public harness methods through Protocols, private attributes through arguments.** The ops
  tools call `HarnessOps` and `WorkflowSlots` methods; the merge tools receive `lane_view`,
  `worker_view`, `journal`, `registry`, `waiters`, `merge_queue`, `event_store`, `file_escalation` as
  explicit arguments from the composition root. The two views are providers because the harness builds
  its worker in `_start_merge_worker`, after it composes the server. No `getattr(harness, '_…')`
  survives (heuristic 7).
- **D5 — the ladder's orchestrator knowledge is one frozen bundle.** `LadderLookups` fields default to
  `None`, each documented with the degradation it reproduces (today's "no orchestrator attached"
  behaviour). The fused-memory harness passes the default.
- **D6 — mutation primitives return a result, not `None`.** `MutationResult(outcome, record)` with
  `outcome ∈ {mutated, unchanged, not_found, not_pending, not_terminal, unparseable}`; each public
  mutator maps it to its existing return shape, so callers see no change.
- **D7 — callback delivery is the queue's job.** One mechanism (heuristic 10), stated once, chosen by
  the registrant with `deliver_on`. The cascade in `park`/`resolve` keeps the `l2-cascade:<id>`
  `resolved_by` encoding, written through its one writer `escalation/models.py::cascade_resolved_by`
  (`343e658060`); S2 (R9) owns the typed `cascade_parent_id` field.
- **D8 — the index is append-only JSONL per task for the archive tier, repaired by the sweep
  (OQ-6, ruled (a) by Leo).** An append is one `O_APPEND` `write()` of one line on a local filesystem, taken after the
  record is durable; appends for different ids hold different per-id locks and rely on that single
  write, not on `PIPE_BUF` (a pipe guarantee). Readers tolerate a trailing partial line. Pruning
  appends tombstones and never rewrites the index, which is what makes "ever filed" hold. Rejected:
  index-only pending reads — each project's queue has two long-running writers, that project's
  orchestrator (which hosts the escalation server) and the single fused-memory service, restarting on
  separate clocks, so one of them can run index-unaware code for a while after a release and hide a
  pending record from the fail-closed veto (B1); a crash between a record and its index line would hide
  it until the next startup sweep even without a release; and nothing records "every writer of this
  directory writes index lines" (the pending read it would speed up costs about 9 ms and is not
  flagged as slow); narrowing the glob to `esc-{task_id}-*` (the esc-3999-2 ruling: 42 live records
  diverge); an in-memory index (two long-running processes and short-lived CLIs write the store);
  sqlite (a second durable store beside the records).
- **D9 — the merge tools' tests move with the tools, through two builders.** The moving test files
  (listed per leaf) go to `orchestrator/tests/`; the ladder tests stay. Each call site migrates by one
  mechanical substitution onto `dark_factory_server` or `ladder_server`, typed against C1/C2/C3 (the
  Tests stance: a fake `_merge_worker` attribute on a `SimpleNamespace` harness pinned a private name).
  Test bodies are not rewritten.
- **D10 — rename, do not re-mint, the dedupe vocabulary.** `dedupe_merge_request` keeps its signature
  through α; ε2 swaps `live_snapshot` for `lane`. `MergeDispatchResult.in_flight` is unchanged; it gains
  `lane_unanswerable` in ε2.
- **D11 — prose rehome per leaf.** Each leaf rewrites the docstrings of the code it moves or edits
  under the Comments stance and retargets the prose pointers (`server.py::<symbol>`,
  `escalation.server.<symbol>`) to the symbols it moves, as 6319 did in `76045318ae`; λ finishes what
  the moves leave.
- **D12 — two primitives, read as R10's one.** R10 says "one locked pending-only `_mutate_pending`
  under every per-record mutator". Two of the eight copies (`patch_resolution_metadata`,
  `note_suppressed_refile`) mutate terminal records and cannot use a pending-only primitive. β reads R10
  as one primitive per record state: `_mutate_pending` under every pending mutator, `_mutate_terminal`
  under the terminal ones, both on one lock-locate-parse-write core.
- **D13 — external lock holders are named, not absorbed.** C4 binds `EscalationQueue` only. The sweep's
  relocation passes are moves, not mutations, and stay. `backlog_policy.py::BacklogPolicy.
  _merge_onto_persisted` and `fused-memory/scripts/backfill_gate_backlog_summaries.py` stay; becoming
  consumers of a public primitive is a later leaf (xrefs to S1 and S12).
- **D14 — one internal enum, one total mapping, inside the orchestrator (OQ-8, superseded by Leo).**
  The lane's `ItemLifecycleState` stays internal; the public vocabulary stays
  `shared.merge_state.MergeState`, which MCP clients, the polling skills and the dashboard switch on.
  `queries.py::LIVE_STATE_OF: Mapping[ItemLifecycleState, MergeState]` (built by S6 μ) is the one
  mapping; a test fails when a non-terminal member has no entry, so an unrecognised state cannot reach
  a caller and nothing passes through raw. Rejected: making the internal enum public (every lane
  refactor would become a public-vocabulary change — tasks 2435 and 2173 already renamed and collapsed
  these states), and keeping strings with a pass-through (its only reason was the layering limit this
  plan removes). `get_merge_queue` keeps rendering `snapshot()` strings for humans; that is a display.

## Contracts

```python
# C1 — orchestrator/src/orchestrator/merge_lane/queries.py (α; S3 owns; S6 C5 implements natively in μ)
@dataclass(frozen=True)
class LiveRequestState:
    request_id: str
    branch: str                       # bare queued-branch id
    task_id: str | None
    state: MergeState                 # a shared.merge_state LIVE_STATES member, via LIVE_STATE_OF; never empty (OQ-8, ruled)
    position: int | None
    enqueued_at: float | None
    verify_started_at: float | None   # None until the verify leg starts (C3's in-verify test)
    waiter_alive: bool

@dataclass(frozen=True)
class TerminalRequestState:
    request_id: str
    state: MergeState                 # a TERMINAL_STATES member
    reason: MergeReason | None
    at: float

class QueryUnavailable(RuntimeError):         # the lane cannot answer now; never a silent False
    query: str                                # the method that raised (INV-2)

class MergeLaneQueries(Protocol):             # stateless; one view per gate or tool call
    def live_request_state(self, request_id: str) -> LiveRequestState | None: ...
    def live_states_for(self, *, branch: str | None = None, task_id: str | None = None) -> tuple[LiveRequestState, ...]: ...
    def owns_merge_worktree(self, path: Path) -> bool: ...
    def terminal_state(self, request_id: str) -> TerminalRequestState | None: ...

LIVE_STATE_OF: Mapping[ItemLifecycleState, MergeState]   # S6 μ adds it here; total over non-terminal members, test-checked (OQ-8, ruled)
# The native implementation (S6 C5 `LaneQueries`) is S6 μ's; no adapter over snapshot() exists (OQ-1, ruled).

# orchestrator/src/orchestrator/merge_lane/types.py (ε2 and ε1)
class MergeDispatchResult:                    # existing; ε2 adds one field
    lane_unanswerable: str | None             # QueryUnavailable.query when the gate fell back to dedupe
class MergeCancelVerdict:                     # existing; ε1 retypes one field
    state: ItemLifecycleState | None          # was the snapshot() wire string; mapped through LIVE_STATE_OF

# C2 — orchestrator/src/orchestrator/mcp/ops_tools.py (δ)
class WorkflowSlots(Protocol):                # S3 owns (OQ-7, ruled); Harness implements until S8's SlotRegistry does
    def is_active(self, task_id: str) -> bool: ...
    def cancel(self, task_id: str) -> bool: ...       # when True, has stamped scheduler.note_workflow_cancelled(task_id) (R3 grace)
    async def wait_cleared(self, task_id: str, timeout_secs: float) -> bool: ...

class HarnessOps(Protocol):
    def task_runtime_snapshot(self) -> list[TaskRuntimeState]: ...
    def halt_merge_queue(self, reason: str) -> dict[str, Any]: ...
    def force_unhalt_merge_queue(self, reason: str) -> dict[str, Any]: ...
    def get_merge_halt_status(self) -> dict[str, Any]: ...
    async def force_halt_scheduler(self, reason: str) -> dict[str, Any]: ...
    async def force_resume_scheduler(self, reason: str) -> dict[str, Any]: ...
    async def reload_config(self) -> dict[str, Any]: ...
    async def clear_verify_host_quarantine(self, host: str) -> dict[str, Any]: ...

def register_ops_tools(mcp: FastMCP, *, ops: HarnessOps, slots: WorkflowSlots, scheduler: Scheduler, git_ops: GitOps | None) -> None: ...

# C7a — orchestrator/src/orchestrator/mcp/merge_tools.py (ε1 creates; ε2 completes)
def register_merge_tools(
    mcp: FastMCP, *, lane_view: Callable[[], MergeLaneQueries | None],    # S6's native implementation; None before the worker starts
    worker_view: Callable[[], SpeculativeMergeWorker | None], merge_queue: asyncio.Queue, event_store: EventStore | None,
    orch_config: OrchestratorConfig, registry: InFlightMergeRegistry, waiters: MergeWaiters,
    journal: MergeQueueStore | None, git_ops: GitOps | None,
    file_escalation: Callable[[Escalation], Awaitable[dict[str, Any]]],
    mint_escalation_id: Callable[[str], str],   # bound to queue.make_id; merge_tools calls it via asyncio.to_thread
    scheduler: Scheduler | None,
) -> None: ...

# escalation/src/escalation/filing_tools.py (ε1 creates with this one function; ζ adds the tools)
async def submit_or_dedupe(queue: EscalationQueue, esc: Escalation, *, dedupe_config: DedupeConfig) -> dict[str, Any]: ...

# C3 — escalation/src/escalation/lookups.py (ζ)
@dataclass(frozen=True)
class LadderLookups:
    task_status: Callable[[str], Awaitable[str | None]] | None = None
    task_claimant: Callable[[str], Awaitable[str | None]] | None = None
    task_statuses: Callable[[Sequence[str]], Awaitable[tuple[dict[str, str], Exception | None]]] | None = None
    workflow_active: Callable[[str], bool] | None = None          # bound to WorkflowSlots.is_active
    routing_tier_bump: Callable[[str], None] | None = None
    granted_files_blocker: Callable[[str], GrantDelivery | None] | None = None
    sha_resolver: CitationResolver | None = None                  # escalation.sha_citations.CitationResolver

# C7b — escalation/src/escalation/server.py (ζ)
def create_server(
    queue: EscalationQueue, *, dedupe_config: DedupeConfig | None = None,
    lookups: LadderLookups = LadderLookups(), startup_sweep: bool = True,
    startup_sweep_now: datetime | None = None, store_identity: StoreIdentity | None = None,
    prefix_resolver: PrefixResolver | None = None,          # task 5324's uuid-prefix guard, unchanged
    mismatch_streak_threshold: int = DEFAULT_MISMATCH_STREAK_THRESHOLD,
) -> FastMCP: ...
def register_filing_tools(mcp, *, queue, dedupe: DedupeConfig, lookups, guard: ProjectRootGuard) -> None: ...
def register_resolve_tools(mcp, *, queue, lookups, guard) -> None: ...
def register_read_tools(mcp, *, queue, lookups, guard) -> None: ...

# C4 — escalation/src/escalation/queue.py (β)
@dataclass(frozen=True)
class MutationResult:
    outcome: Literal['mutated', 'unchanged', 'not_found', 'not_pending', 'not_terminal', 'unparseable']
    record: Escalation | None

class EscalationQueue:
    def _mutate_pending(self, escalation_id: str, mutate: Callable[[Escalation], bool], *, archive_after: bool = False) -> MutationResult: ...
    def _mutate_terminal(self, escalation_id: str, mutate: Callable[[Escalation], bool]) -> MutationResult: ...
    def set_resolve_callback(self, callback: Callable[[Escalation], None], *, deliver_on: asyncio.AbstractEventLoop | None = None) -> None: ...
    def set_notify_callback(self, callback: Callable[[Escalation], None], *, deliver_on: asyncio.AbstractEventLoop | None = None) -> None: ...

# C5 — escalation/src/escalation/queue.py (θ); index file data/escalations/by-task/<encoded>.jsonl
#   <encoded> = 't-' + urllib.parse.quote(task_id, safe='')  ('/' never appears; '.'/'..' become
#   ordinary names); over 200 bytes → 'h-' + sha256(task_id).hexdigest(). Each line carries the real task_id.
@dataclass(frozen=True)
class IndexEntry:
    id: str; task_id: str; filed_at: str; level: int; category: str; severity: str
@dataclass(frozen=True)
class IndexTombstone:                 # written by prune_archive from the record it parsed before rmtree
    id: str; task_id: str; pruned_at: str; filed_at: str; level: int; category: str; severity: str
@dataclass(frozen=True)
class TaskHistory:
    records: tuple[Escalation, ...]
    pruned: tuple[IndexTombstone, ...]   # built from the task's tombstones; joined by id to its IndexEntry
                                         # only when one exists; a tombstone with no IndexEntry is reported as-is,
                                         # and `level=` filters on the tombstone's own level
class EscalationQueue:
    def get_by_task(self, task_id: str, *, status=None, level=None, agent_role=None) -> list[Escalation]: ...  # root glob ∪ indexed archive
    def get_task_history(self, task_id: str, *, level: int | None = None) -> TaskHistory: ...

# C6 — escalation/src/escalation/models.py (κ), beside HARNESS_SENTINEL_ROLE_PREFIXES
LEVELS_HEADER = 'X-Escalation-Levels'
IDENTITY_HEADER = 'X-Escalation-Identity'
WATCHER_AUTO_IDENTITY = 'orchestrator-escalation-watcher-auto'
@dataclass(frozen=True)
class CallerHeaders:
    levels_raw: str | None; identity: str | None
def read_caller_headers(headers: Mapping[str, str]) -> CallerHeaders: ...   # looks up the .lower() names
# escalation/src/escalation/authority.py
def watcher_connection_headers() -> Mapping[str, str]: ...   # derived from ROLE_LEVEL_ALLOWLIST

# C8 — escalation/src/escalation/ladder_gates.py (ι)
@dataclass(frozen=True)
class GateInput: ...                          # every parsed field the gates read; tactical beyond frozen
@dataclass(frozen=True)
class Refusal:
    gate: str; code: str; facts: Mapping[str, Any]          # INV-2
Gate = Callable[[GateInput], Refusal | None]
RESOLVE_GATES: tuple[Gate, ...]; PROMOTE_GATES: tuple[Gate, ...]
def first_refusal(gates: Sequence[Gate], ctx: GateInput) -> Refusal | None: ...
```

Invariants. C1: a `MergeLaneQueries` answers from the lane's own state in O(live items), runs no
audit and holds no state across calls; a gate or tool calls `lane_view()` once per call; `None` from a
lookup means "not live/not known", `QueryUnavailable` means "could not answer"; `state` is always a
`MergeState`, because every non-terminal `ItemLifecycleState` has exactly one `LIVE_STATE_OF` entry
and a test fails otherwise. C2/C3: every method and field maps to one public harness method; a `None`
field reproduces the documented degradation and nothing else. C4: no `EscalationQueue` method other
than `_mutate_pending`, `_mutate_terminal` and the creators' shared lock-and-write helper writes a
record; `_mutate_pending` reads the root only (a pending record lives in the root by construction); a
callback registered with `deliver_on` runs on that loop's thread, after the lock is released, in
submission order, and a closed loop is a counted, logged drop, never a raise; external lock holders are
D13's. C5: the record is durable before its index line is appended; every record `submit` or
`submit_resolved` writes has an index line before the call returns; after pass 0 of the sweep every
record on disk with a non-`None` `task_id` has one (records with `task_id=None` — 426 measured by task
4451 — are not indexed, and no per-task read can ask for them); a pending record is returned by
`get_by_task`/`has_open_l1` whether or not it has an index line; the archive tier touches no archive
file that is not the task's own; a pruned id has a tombstone. `prune_archive` parses each record of a
dated dir before its `rmtree` to find its task's index file; an unparseable record is a counted WARNING
and gets a tombstone under its id-only name (`by-task/` file `u-<id>.jsonl`, carrying `id` and
`pruned_at` only, never joined to a task). A pruned record with no index line is still reported by
`get_task_history` and the history tools, at every `level=`, from its tombstone's parsed fields. C6: no module spells a header name or
the watcher identity as a literal outside `models.py` (scripts included), and no module but
`read_caller_headers` reads the headers (S2's invariant will scan for it). C8: `resolve_issue`'s steps
(the 12 listed by fk-609a1832903b plus the recut and identity-policy gates) and `promote_to_l2`'s steps
are each classified separately; gates are pure over `GateInput`; the order of
`RESOLVE_GATES`/`PROMOTE_GATES` is the order refusals are reported. Error semantics otherwise
unchanged from today's tools; exact field names beyond those above are tactical.

## Boundary-test sketch

Suite per row (μ): escalation — rows 4–8, 10–14, 16, 17; orchestrator — rows 1–3, 9, 15.

| # | Side | Scenario | Pre | Post |
|---|---|---|---|---|
| 1 | ε2 lane | `dedupe_merge_request` with `lane_view` returning S6's native implementation over a REAL `SpeculativeMergeWorker` holding one queued and one verifying entry | real worker fixture | resubmit of the queued branch → REPLACE/dedupe as C3 says; of the verifying branch → `duplicate_in_verify` (SUPERSET) or dedupe (SAME/SUBSET, D1); `snapshot()` is never called by the gate |
| 2 | ε2 lane | the port raises `QueryUnavailable` | injected port that raises | gate dedupes, emits the recorded fail-safe event, and `merge_request`'s response carries `lane_unanswerable: '<query>'` |
| 3 | ε1 + S6 μ | every non-terminal `ItemLifecycleState` member (REDISPATCH_PARKED and DISPATCHING included) | parametrised over the enum | `LIVE_STATE_OF` has exactly one `MergeState` entry per member (the exhaustiveness test, which replaces the pinned pass-through test and closes 6827); for a live request in each state `merge_status` answers that `MergeState`, and `merge_cancel`'s refusal on it answers the same; adding an enum member without an entry fails the test |
| 4 | β queue | two processes `amend` the same pending record concurrently | real queue dir | both amendments present; one write per mutation; no lost update |
| 5 | β queue | `stamp_triage` on an archived record | resolved record | `not_pending`, no write, no root file resurrected |
| 6 | β queue | `resolve()` called from a worker thread with `deliver_on=loop` | running loop, thread | callback runs on the loop thread after the lock; `asyncio.Event.set()` wakes an awaiting coroutine; `resolve()` returned before the callback ran |
| 7 | β queue | `resolve()` with no `deliver_on`; and with `deliver_on` a closed loop | — | inline on the caller's thread (documented default); closed loop → WARNING, drop counted, `resolve()` returns normally |
| 8 | ζ server | `create_server(queue, store_identity=..., prefix_resolver=...)` (the fused-memory shape) | no lookups | 13 tools listed, none merge/ops; `escalate_info` + `resolve_issue` round-trip over HTTP; `fused-memory/tests/test_stages.py::test_every_escalation_server_tool_is_classified` passes with no merge/ops name in any bucket |
| 9 | ζ+ε+δ server | dark-factory shape: `dark_factory_server(...)` (`create_server` then `register_merge_tools`, `register_ops_tools`) | fakes typed against C1/C2 | 29 tools under today's names; `merge_request` → `dedupe_merge_request` path; `release_workflow` waits through `WorkflowSlots.wait_cleared`; `merge_cancel` on a workflowless request files through `file_escalation`, the filed id is `esc-<task_id>-<n>` (minted by `mint_escalation_id`), and the record is deduped like any ladder filing |
| 10 | ζ layering | AST walk of `escalation/src` | — | zero `orchestrator` imports at any depth; `server.py` has no `@mcp.tool` decorator; a seeded fixture module with a function-local import fails the walk |
| 11 | ζ ladder | `get_pending_escalations` with and without `LadderLookups.task_statuses` | both | `pins_recovery` present only when the lookup is wired (today's contract) |
| 12 | θ index | `submit` then `get_by_task(status='pending')`, resolve, archive, `get_by_task()` default | real queue dir | index line written after the record and before `submit` returns; default read returns the archived record with zero archive rglob (patched `_iter_archive_paths` asserts zero calls) |
| 12b | θ index | a pending record written by an index-unaware writer (`_atomic_write`, no append) | real queue dir | returned by `get_by_task(status='pending')` and `has_open_l1`; indexed by the next pass 0 |
| 13 | θ index | records dropped into the root/archive by hand, then `run_startup_sweep`; a task_id containing `/` and `..` | foreign-written files | pass 0 indexes them; a divergent-id record (`esc-curator-N`, `task_id='task-curator'`) is found by its task_id; the encoded filename stays inside `by-task/` |
| 14 | θ history | a dated dir pruned by `prune_archive` (and by `archive.main`); one pruned record that never had an index line (archived by an index-unaware writer, pruned before pass 0) | indexed records + one unindexed | tombstones appended with the parsed `filed_at`/`level`/`category`/`severity`; `get_task_escalation_history` → `count: 0, pruned_count: N, pruned: [...]`; `get_task_escalations` the same envelope; the unindexed pruned record is still reported when read with `level=` set to its level; a record moved mid-read is not reported pruned |
| 15 | 5409 events | `merge_request(verified_green=True)` after ε2 moved it | event store | one `workflow_verify` row with `tip_sha` and `runner`, built by 5409's shared constructor; `green_checkpoint_at_tip` returns it (asserted by 5409 before the move; μ replays it after) |
| 16 | κ headers | `watcher_connection_headers()` (title-case) over real HTTP vs `resolve_issue` on L2 | one constant set | the supervised watcher's headers are refused at L2 (`level_forbidden`) and accepted at L1 — the cross-layer pin the harness comment says no test exercises |
| 17 | ι gates | an extra gate inserted into `RESOLVE_GATES` by a test; a ceiling-denied L2 close that matches the carve-out | injected tuple | the refusal surfaces with the gate's name; the other gates run in order; the carve-out admits only when the ceiling denies; the park branch is chosen before any gate |

## Cross-PRD relationship

| Other stream / PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| S6 merge-lane lifecycle | produces (to S6) / consumes (from S6) | `MergeLaneQueries` (C1) defined in `merge_lane/queries.py` by α. S6's native-port task μ builds the only implementation (S6 C5 `LaneQueries`, over its lifecycle registry and the journal) and the `LIVE_STATE_OF` table with its exhaustiveness test in that file (OQ-1 ruled (a), OQ-8 superseded); no adapter exists anywhere. Edges: S6 μ → S3 α; S3 ε1 → S6 μ and S3 ε2 → S6 μ; S6 μ must NOT depend on any S3 merge-tool move (that closes a cycle). ε1 wires S6's class into `harness.py` (S6 has no adapter composition to replace) and retypes `MergeCancelVerdict.state` in S6's `types.py` and its producer in `worker.py`. Same-file orders (the `harness.py` one ruled by the lead 2026-11-10): `intake.py` S3 α → S6 lifecycle (ζ) and limits (κ) → S3 ε2; `harness.py` — S6 α first (never held behind S3), then S3 β, δ, κ, then S6 β1 … μ, then S3 ε1, ε2, ζ (edges: S3 β → S6 α; S6 β1 → S3 κ). `intake.py`'s `retention=` removal is task 4833's, which lands first in today's files: before S3 ε1 and before S6 ζ and κ (S6 κ → 4833); 4833 waits on no S3 or S6 task. Comment mentions of the old gate name in `worker.py`/`queue_store.py`/`tip_relation.py` are S6's; S6's C7 `HaltState` owner semantics reach δ's halt/unhalt tools (C2 unchanged now) | S3 port, S6 implementation and table | lead: edges at decompose |
| S7 merge-lane gates | consumes (from S7) | S7-3 ruled (b): S7's deletion task (δ) removes `merge_request`'s `retry_failed_only` parameter, its docstring paragraph and its threading call from today's `server.py`, with the producer in `worker.py`; ε2 depends on S7 δ, so the move carries one parameter fewer. S7 δ is first in S7's chain (lead, 2026-11-10); S6 α takes no edge on S7 δ or S7 β (S7 δ → S6 α and S7 β → S6 α where they share a file), so S3 β … ι do not wait on any S7 task. On `server.py` S7 δ is region-disjoint from S3 α, δ, κ, θ, ι and the lead's per-file order places it; it must not depend on S3 ε2 or any later S3 leaf (cycle) | S7 | lead: edge ε2 → S7 δ |
| S8 harness composition root | produces (to S8) | `WorkflowSlots` (C2) in `orchestrator/mcp/ops_tools.py` (OQ-7, ruled (a)): S8's `SlotRegistry` implements it (its `cancel` keeps the `scheduler.note_workflow_cancelled` stamp, so S8 binds a wrapper, not bare `SlotRegistry.cancel`), S8 drops its own C1 `WorkflowSlots` definition and deletes `Harness.is_workflow_active`/`cancel_workflow` forwarders; `LadderLookups.workflow_active` bound to `slots.is_active`. Cycle cut confirmed by the lead (2026-11-10): S8 α depends on S3 δ; S8 β depends only on S3 β and S3 δ (never on S3 ε1, ε2, ζ); S3 ε1 depends on S8 ε2b (per-file order on `harness.py`); S8 ε3 depends on S3 ζ; S8 ω deletes the `Harness.is_workflow_active` forwarder and depends on S3 ζ. S3 κ precedes S8 β transitively (S8 β → S6 κ → … → S6 β1 → S3 κ). Task 4651's `escalation.git_authority` import lives in S8's `stranded_reconcile.py` (4651 → S8 ε2a, S8-2), where ε1 repoints it | S3 port, S8 implementation | lead: edges at decompose |
| S2 typed record facts | produces (to S2) | C6 header constants and `read_caller_headers` in `models.py`, beside `HARNESS_SENTINEL_ROLE_PREFIXES` (task 5224) and the cascade writer/reader `cascade_resolved_by`/`cascade_parent_id` (`343e658060`); `RESOLVE_ACTIONS`/`_DISMISS_ACTIONS` and the five header-reading tools in `resolve_tools.py` after ζ; β keeps the `l2-cascade:` encoding; the stored category `merge_coalesce_live_original` and the sentinel task_id prefix `'__merge_coalesce_live_original__'` keep their values | S3 lands first; S2 consumes | S2 wave 3 |
| S1 filing entry point | produces (to S1) | `filing_tools.py` hosts `submit_or_dedupe` (ε1) and `_chokepoint_or_submit` (ζ) until S1's `filing.py` takes them; R8's `filing.file` later replaces the `file_escalation` binding | S1 | ι does not touch the chokepoint |
| S5 GitOps ports | produces (to S5) | δ repoints `release_warm_worktree` onto `shared.git_async.run_git` while moving it (S5 C1's table assigns that edit to S3 δ); S5 β (deletes `orchestrator.git_ops._run`) depends on S3 δ, which is not held; S5 ξ0/λ/ο0 edit `claim_warm_worktree`/`release_warm_worktree`/`cleanup_merge_worktree`/`advance_main` callers in whichever module holds them — whichever of S5's leaf and S3 δ/ε1/ε2 lands second adapts (S5's rule, adopted) | S5 | edge S5 β → S3 δ at decompose |
| S10 task-write contract | adjacent | S10 β1/β2 edit `merge_relane`'s `_write_merge_lane_metadata`, which ε1 moves (S10's "S3 ε" means ε1). S10 is filed first and ε1 is held, so ε1 depends on β1/β2 and carries their edit | S10 | lead: edge ε1 → S10 β1/β2 |
| S9 workflow spine | adjacent | S9 κ edits `workflow.py::_emit_workflow_verify`, which 5409 turns into a caller of its shared constructor | S9 / 5409 | 5409 and S9 κ serialized; α edits `workflow.py` (`_submit_to_merge_queue` event) — same-file edge |
| S4 verify | adjacent | 5409 adds `workflow_verify_event_data` to `verify_evidence.py` | S4 / 5409 | same-file edge if S4 touches it |
| S4, S6, S7, S8 `event_store.py` leaves | adjacent | α adds `EventType.merge_deduplicated` | each | α and those leaves serialized |
| S12 / S1 external lock holders | produces (to) | `_merge_onto_persisted` and the backfill script may adopt a public primitive later (D13) | S12 / S1 | xref |
| S22 dashboard, S25 scripts | produces (to) | γ repoints `iter_all_escalation_paths`/`read_escalation_for_scan` importers in dashboard (2 files + 1 prose), scripts (4 + `escalation_ladder.py`), fused-memory (1), orchestrator (1); κ repoints `scripts/sitting/apply.py::IDENTITY_HEADER` | S3 edits the import lines only | none |
| fused-memory recon stage classification | produces (to) | δ, ε1, ε2 drop each moved tool from `fused-memory/tests/test_stages.py::_REVIEWED_STAGE_SAFE`; ε1 empties `cli_stage_runner.py::DISALLOW_ESCALATION_MERGE_LANE_CONTROL` (its only member, `merge_relane`, is no longer served on :8103) and updates the test that pins it | S3 | none |
| S19 layering | produces (to) | `test_layering.py` is the escalation instance of the DAG invariant S19 promotes | S19 may generalise | none |
| Task 5409 (pending) | owns `fk-bc64099a1cd0` | the `workflow_verify` constructor; lands FIRST in today's `server.py`, `workflow.py` and `verify_evidence.py` (OQ-1 ruling) and absorbs `runner`; ε2 depends on it and carries its call | 5409 | lead: amend, edge ε2 → 5409 |
| Task 6670 (done, `f6a4482199`) | related | added `runner` to the workflow-side green through `_emit_workflow_verify`; 5409's constructor absorbs it | — | none |
| Task 4833 (pending) | owns `fk-66746efddd4d` | deletes `server.py::_get_terminal_retention`, its three `retention=` sites, the lane-side `retention=` threads and `TerminalOutcomeRetention`; lands FIRST in today's files (lead ruling 2026-11-10, reversing the earlier 4833 → ε2); ε1 depends on it, so both merge-tool moves carry its result; S6's limits task κ (and lifecycle task ζ) also depend on it. Shares `intake.py`/`types.py`/`server.py` with α: 4833 never waits on α; whether α waits on 4833 is the lead's per-file order | 4833 | lead: amend (key, paths), edge ε1 → 4833 |
| Task 4651 (pending, high) | adjacent | imports `escalation.git_authority` (that module's docstring); lands FIRST against today's path (OQ-1 ruling), re-scoped onto S8's `StrandedReconciler` (depends on S8 ε2a), so the import sits in `stranded_reconcile.py`; ε1 moves the module and repoints the import there | 4651 | lead: edge ε1 → 4651 |
| Task 4567 (pending, high; 8 commits on `task/4567`) | adjacent | edits `queue.py`'s `get_by_task` pre-scan and `_locate_path` memo; moves the in-process archive-inclusive calls off the loop | 4567 | lead: edge θ → 4567 (θ only, not β — lead ruling 2026-11-10) |
| Task 6571 (pending) | subsumed | tier walk extraction into escalation — contradicts R10 | ε1 | lead: cancel at decompose, `x_acceptance_reason` names ε1 |
| Task 6827 (pending) | covered | `_map_live_state` gaps; the total `LIVE_STATE_OF` table maps REDISPATCH_PARKED (wire `awaiting_host`) and DISPATCHING, and `passthrough` no longer exists (OQ-8, ruled) | S6 μ (table) + ε1 (deletes `_map_live_state`) | lead: cancel at decompose as covered |
| Task 6004 (pending, low) | related | a bulk archive-inclusive L2 read; θ removes its premise | 6004 | lead: note |
| Task 3176 (pending) | adjacent | edits `harness.py::_WATCHER_ALLOWED_TOOLS`, same file as κ | 3176 | same-file edge at decompose |
| Tasks 8422, 5324, 5516, 5224, 8336 (done) | context | added `claim_policy_class` + `policy_gate.py`, the uuid-prefix guard (`prefix_resolver`), the recut gate, `HARNESS_SENTINEL_ROLE_PREFIXES` in models.py, `git_authority`'s `Scheduler.read_task` read; each is carried by the leaf that moves or edits its code | — | none |
| `review/briefing.yaml` known gap "duck-typed harness._merge_worker" | closes | removed by ε2 | ε2 | — |

## Decomposition plan

Overlay bands: authored deltas are within 300–1,500 LOC. δ, ε1, ε2, ζ and γ carry moved code or
mechanical import/call-site substitutions beyond that and exceed 15 declared files; each crosses the
overlay's review trigger, examined: δ/ε1/ε2/ζ files are the moved tools' own tests migrating by one
substitution onto a builder, and γ's are one-line import repoints; splitting a move further would
stitch one unit across leaves. Arrows below read "lands before". Same-file leaves are serialized by
real edges: `queue.py` β → γ → θ → λ, with 4567 → θ; `server.py` α → δ → κ → θ → ι → ε1 → ε2 → ζ
(4833 → ε1; S7 δ → ε2, and S7 δ against α … ι is region-disjoint and the lead orders it; S10 β1/β2 → ε1; S10 β1/β2 against
α … ι are region-disjoint and the lead orders them); `harness.py` S6 α → β → δ → κ → S6 β1 … S6 μ →
S8 ε2b → ε1 → ε2 → ζ → S8 ε3 (lead rulings 2026-11-10), interleaved by the lead with the other S5, S8,
S9 and S10 `harness.py` leaves; `intake.py` α → S6 lifecycle → S6 limits → ε2 (4833 also edits it,
lands first in today's files and before ε1, and waits on no S3 or S6 task);
`merge_lane/types.py` S6 α → α → (S6's other `types.py` tasks) → ε1 → ε2; `models.py` γ → κ; `filing_tools.py` ε1 →
ζ; `resolve_tools.py`/`read_tools.py` ζ → λ; `merge_tools.py` ε1 → ε2 → λ;
`fused-memory/tests/test_stages.py` δ → ε1 → ε2. `high` leaves (β, δ, ε1, ε2, ζ, μ) become
dependencies of gate 6698.

Held: ε1 and ε2 depend directly on S6 μ (the native port and `LIVE_STATE_OF`); ζ, λ and μ wait
through ε2. S6 μ's priority is S6's to set and is not raised by hand: the scheduler's priority
inheritance (`Scheduler._compute_effective_priorities`) runs it, and S6's chain beneath it, at the
highest priority among its undone dependents once ε1/ε2 (high) are filed. Not held on S6 μ: α, β, γ,
δ, θ, ι, κ; of these, β (and through it γ, δ, κ, θ, ι) waits on S6 α, which inherits β's high priority
the same way, and θ also waits on task 4567.

**α — `MergeLaneQueries` port and records; `dedupe_merge_request` rename**
(`task_kind=normal`, priority medium, est. ~400 LOC + mechanical rename across ~20 test files)
- Files: `orchestrator/src/orchestrator/merge_lane/queries.py` (new: the Protocol, `LiveRequestState`,
  `TerminalRequestState`, `QueryUnavailable`), `orchestrator/src/orchestrator/merge_lane/intake.py`
  (rename only), `orchestrator/src/orchestrator/merge_lane/intake_buffers.py`,
  `orchestrator/src/orchestrator/merge_lane/__init__.py`, `orchestrator/src/orchestrator/merge_lane/types.py`
  (`record_coalesced`, `coalesced_request_ids`), `orchestrator/src/orchestrator/event_store.py`,
  `orchestrator/src/orchestrator/workflow.py` (the `emit_merge_coalesced` import and
  `_submit_to_merge_queue`'s event), `escalation/src/escalation/server.py` (the rename at
  `merge_request`'s import and call), `orchestrator/tests/test_merge_lane_queries.py` (new),
  `orchestrator/tests/test_merge_lane_intake.py`, `orchestrator/tests/test_merge_queue_c3_submit_identity.py`,
  `orchestrator/tests/test_merge_queue.py`, `orchestrator/tests/merge_lane_ratchet_baseline.json` and one
  new `orchestrator/tests/merge_lane_ratchet_raises/<α>-*.json` record (`--authorize-raise`), plus the
  rename sweep.
- Keys: `fk-c8c3406fd7f4` (the port; ε2 finishes it), `fk-0de72a91f89a`. Deletes: the dedupe-sense
  "coalesce" identifiers (renamed); nothing of the dict parsing yet.
- Deps: none. Seams: produces C1 for S6 μ (S6 μ depends on α) and for ε1/ε2; `event_store.py`
  serialized with S4/S6/S7/S8 leaves.
- Signal: `git grep 'coalesce_or_enqueue_merge_request('` over `orchestrator/src` and
  `escalation/src` is empty; a live `merge_request` resubmission on dark-factory logs
  `dedupe_merge_request: …` and writes a `merge_deduplicated` event.
- G7: INV-1 (one typed Protocol both sides check against), INV-2 (the `QueryUnavailable` message names
  request_id and the raising query).

**β — `_mutate_pending`/`_mutate_terminal` under every mutator; callback delivery contract**
(`task_kind=normal`, priority high, est. ~750 LOC)
- Files: `escalation/src/escalation/queue.py`, `orchestrator/src/orchestrator/harness.py` (the two
  `set_*_callback` registrations gain `deliver_on=asyncio.get_running_loop()`), `escalation/tests/test_queue.py`,
  `escalation/tests/test_queue_mutation.py` (new), `orchestrator/tests/test_harness_escalation_callbacks.py` (new).
- Keys: `fk-4a0626fc033d`, `fk-07be43f313bc`. Deletes: 8 RMW copies (task 8422's `claim_policy_class`
  included), `_load_root_pending` as a separate helper, the `get()`-under-lock variants,
  `_write_late_resolution`'s own write, every `(json.JSONDecodeError, KeyError, TypeError)` catch
  outside the primitive core.
- Deps: S6 α (`harness.py`; lead order 2026-11-10 — S6 α is never held behind S3). Not 4567 (θ's
  prerequisite only). Seams: C4; D7 keeps `l2-cascade:` for S2; D13 names the external lock holders;
  S8 β depends on β.
- Signal: boundary rows 4–7. In the live orchestrator, `resolve_issue` from an MCP client logs the
  workflow wake on the loop thread (the `_schedule_coro_threadsafe` off-loop branch no longer fires for
  escalation callbacks).
- G7: INV-8 (callback delivery is `call_soon_threadsafe`, no loop-blocking I/O added), INV-3 (the
  primitive re-reads under the lock before writing), INV-11 (a closed-loop drop is logged and counted).

**γ — `scan.py`, late-resolution policy into `amendments.py`, importers repointed, queue prose**
(`task_kind=normal`, priority medium, est. ~500 LOC)
- Files: `escalation/src/escalation/scan.py` (new), `escalation/src/escalation/amendments.py`,
  `escalation/src/escalation/queue.py`, `escalation/src/escalation/models.py` (pointer prose),
  `escalation/src/escalation/sweep.py`, `escalation/src/escalation/watcher.py`,
  `escalation/src/escalation/shadow_ruling.py`, `dashboard/src/dashboard/data/escalation_corpus.py`,
  `dashboard/src/dashboard/data/performance.py`, `dashboard/src/dashboard/data/escalations.py` (prose),
  `fused-memory/src/fused_memory/reconciliation/escalation_archive.py`,
  `orchestrator/src/orchestrator/digest.py`, `scripts/legibility/plan_deviation_census.py`,
  `scripts/offline_lane_soak_inputs.py`, `scripts/sitting/inventory.py`, `scripts/sitting/fleet_state.py`,
  `scripts/escalation_ladder.py` (prose), `scripts/tests/test_escalation_ladder.py` (prose),
  `escalation/tests/_scan_race_helpers.py` (prose), `escalation/tests/test_note_to_file.py`,
  `escalation/tests/test_resolve_escalation_shas_tool.py`, `escalation/tests/test_shadow_ruling.py`,
  `escalation/tests/test_scan.py` (new), `escalation/tests/test_queue.py`. 24 declared files, past
  the 15-file review trigger: 15 of them are one-line import or pointer edits.
- Keys: `fk-5a9e713f4055`. Deletes: the 143-line inventory docstring; 11 production and 4 test
  `from escalation.queue import …` lines naming a scan helper become `from escalation.scan import …`.
- Deps: β (queue.py). Seams: S22/S25 import lines.
- Signal: `python -c "import escalation.scan"` imports only `escalation.models`, `escalation.classify`,
  `escalation.archive`; `git grep 'ALREADY GUARDED elsewhere'` is empty; the dashboard escalation panel
  and `scripts/sitting/inventory.py` run unchanged against the live store.
- G7: INV-9 (the call-site inventory has no code home; this PRD records the count once).

**δ — operator tools registered from `orchestrator/mcp/ops_tools.py`; `HarnessOps`, `WorkflowSlots`**
(`task_kind=normal`, priority high, est. ~500 LOC authored; ~700 moved incl. tests)
- Files: `orchestrator/src/orchestrator/mcp/ops_tools.py` (new), `escalation/src/escalation/server.py`,
  `orchestrator/src/orchestrator/harness.py` (implements `WorkflowSlots`; composes `register_ops_tools`),
  `escalation/src/escalation/server_instructions.py`, `orchestrator/tests/_escalation_server_factory.py` (new),
  `orchestrator/tests/test_ops_tools.py` (new; absorbs `escalation/tests/test_release_workflow.py`,
  `test_warm_worktree_verbs.py`, `test_clear_verify_host_quarantine.py`, `test_task_runtime_state_tool.py`,
  `test_reload_config_integration.py`, which are deleted), `orchestrator/tests/test_interactive_warm_worktree_integration_gate.py`,
  `orchestrator/tests/test_task_runtime_snapshot.py`, `fused-memory/tests/test_stages.py` (the eleven
  names leave `_REVIEWED_STAGE_SAFE`).
- Keys: `fk-a8d3003e9d83` (part), `fk-e903cc072b15` (the `claim_warm_worktree` and
  `release_warm_worktree` imports). Deletes: 2 function-local imports; `harness.scheduler`/
  `harness.git_ops` attribute reads by the ops tools; `release_workflow`'s poll loop.
- Deps: α (server.py), β (harness.py). Seams: C2 (S8 implements `WorkflowSlots`); S5 β depends on δ;
  S6 C7 halt semantics noted, C2 unchanged.
- Signal: boundary row 9 (ops half); `mcp__escalation__release_workflow` on a live idle task answers as
  today; `release_warm_worktree` removes an `_iact-*` worktree through `shared.git_async.run_git`;
  `fused-memory/tests/test_stages.py::test_every_escalation_server_tool_is_classified` passes.
- G7: INV-13 n/a; INV-11 (no new fail-soft); INV-9 (one `WorkflowSlots` definition).

**θ — per-task landing index for the archive tier; honest history envelopes**
(`task_kind=normal`, priority medium, est. ~650 LOC)
- Files: `escalation/src/escalation/queue.py`, `escalation/src/escalation/sweep.py`,
  `escalation/src/escalation/archive.py` (tombstones), `escalation/src/escalation/server.py` (the
  envelopes of `get_task_escalations` and `get_task_escalation_history`, where they live until ζ),
  `escalation/tests/test_queue.py`, `escalation/tests/test_landing_index.py` (new),
  `escalation/tests/test_sweep.py`, `escalation/tests/test_archive.py`,
  `skills/escalation-watcher-auto/SKILL.md` (the `count: 0` guidance).
- Keys: `fk-e5d81cc1f1d4`, `fk-ce2dc99e0305`. Deletes: the archive rglob and `id_to_paths` pre-scan in
  `get_by_task`'s default read.
- Deps: γ (queue.py), κ (server.py), 4567 (`queue.py`, mid-flight on `task/4567`; θ only — lead
  ruling 2026-11-10). Not held on S6 μ (OQ-1 ruling: placed ahead of the ladder split). Seams: C5;
  S22/S25 readers may adopt `get_task_history`.
- Signal: boundary rows 12–14; on the live store `get_by_task('<task>')` default completes in under 30
  ms (root glob 8.4–9.7 ms plus the task's indexed archive records, vs 316–411 ms measured at
  e4b333a601) and `get_task_escalation_history` for a task with records older than 30 days returns
  `pruned_count > 0`.
- G7: INV-13 (the history tool distinguishes "pruned" from "never filed"; backfill decision per record
  kind), INV-3 (the fail-closed pending veto reads the root, its ground truth; indexed reads go through
  `get()`), INV-7 (the index is owned by the queue, repaired by the sweep, never by a reader), INV-8
  (archive reads are O(task records)).

**ι — `RESOLVE_GATES` and `PROMOTE_GATES` as ordered gate tables**
(`task_kind=normal`, priority medium, est. ~950 LOC)
- Files: `escalation/src/escalation/ladder_gates.py` (new), `escalation/src/escalation/server.py`
  (`resolve_issue`, `promote_to_l2`), `escalation/tests/test_ladder_gates.py` (new),
  `escalation/tests/test_server.py`, `escalation/tests/test_resolve_resolution_action.py`.
- Keys: `fk-609a1832903b` (the two tools; `_chokepoint_or_submit` → S1). Deletes: the inline gate
  chains of both tools, task 5516's recut check and task 8422's identity-policy refusal included.
- Deps: θ (server.py; through it κ, whose `read_caller_headers` the parse step uses). Not held on S6 μ
  (OQ-1 ruling: placed ahead of the ladder split). Seams: C8; S1's filing gates untouched; S2's
  `ACTION_EFFECTS.record_disposition` will branch inside the effect step.
- Signal: boundary row 17; `complexipy` per-function ceiling: `resolve_issue` and `promote_to_l2` ≤ 15
  with the module total not rising (`docs/code-quality.md` §What to measure).
- G7: INV-2 (every refusal names its gate, code and facts).

**κ — header names, the one header reader and the watcher identity in `escalation/models.py`**
(`task_kind=normal`, priority low, est. ~300 LOC)
- Files: `escalation/src/escalation/models.py`, `escalation/src/escalation/authority.py`,
  `escalation/src/escalation/classify.py`, `escalation/src/escalation/server.py` (the five tools'
  header reads), `orchestrator/src/orchestrator/harness.py`, `scripts/sitting/apply.py`,
  `escalation/tests/_escalation_http.py`, `escalation/tests/test_authority.py`,
  `escalation/tests/test_capability_guard_http.py`, `escalation/tests/test_escalation_http_helper.py`,
  `orchestrator/tests/test_harness_watcher_supervisor.py`.
- Keys: `fk-43bad4e724da`. Deletes: `_WATCHER_ESCALATION_HEADERS`, `_LEVELS_HEADER`, `_IDENTITY_HEADER`,
  `_WATCHER_AUTO_IDENTITY`, `scripts/sitting/apply.py::IDENTITY_HEADER`, the `'0,1'` literal, two "must
  stay in lockstep" comments.
- Deps: δ (server.py, harness.py), γ (models.py). Not held on S6 μ (OQ-1 ruling: placed ahead of the
  ladder split, and first of κ, θ, ι on `server.py` so S6's `harness.py` chain, which follows κ, never
  waits on 4567). Seams: C6 for S2; S6 β1 depends on κ (`harness.py`); task 3176 same-file edge on
  `harness.py`.
- Signal: boundary row 16 over real HTTP with the title-case headers (the cross-layer test the
  harness comment says is missing).
- G7: INV-5, INV-9.

**ε1 — merge read/cancel/relane tools from `orchestrator/mcp/merge_tools.py`; three modules move**
(`task_kind=normal`, priority high, est. ~550 LOC authored; ~1,200 tool lines + 1,029 module lines
moved, test files migrated by substitution)
- Files: `orchestrator/src/orchestrator/mcp/merge_tools.py` (new),
  `orchestrator/src/orchestrator/mcp/merge_status_answer.py` (moved from `escalation/src/escalation/merge_status_answer.py`),
  `orchestrator/src/orchestrator/mcp/git_authority.py` (moved from `escalation/src/escalation/git_authority.py`) (OQ-4, ruled),
  `orchestrator/src/orchestrator/merge_lane/lane_resolution.py` (moved from `escalation/src/escalation/merge_lane_resolution.py`),
  `escalation/src/escalation/filing_tools.py` (new: `submit_or_dedupe`), `escalation/src/escalation/server.py`,
  `orchestrator/src/orchestrator/harness.py` (constructs registry + waiters; wires S6's native port as
  `lane_view`; composes `register_merge_tools`; `git_authority` prose pointers),
  `orchestrator/src/orchestrator/stranded_reconcile.py` (S8's module; repoints task 4651's
  `git_authority` import),
  `orchestrator/src/orchestrator/merge_lane/types.py` and `orchestrator/src/orchestrator/merge_lane/worker.py`
  (`MergeCancelVerdict.state` typed, its one producer), `shared/src/shared/task_metadata.py` and
  `shared/src/shared/merge_state.py` (docstring pointers only),
  `fused-memory/src/fused_memory/reconciliation/cli_stage_runner.py` and `fused-memory/tests/test_stages.py`
  (stage classification), tests moved to `orchestrator/tests/`: `escalation/tests/test_merge_relane.py`,
  `test_merge_state_wiring.py` (its pass-through test deleted), `test_merge_status_boundary.py`,
  `test_merge_status_git_authority.py`, `test_merge_status_tiers.py`, `test_merge_lane_resolution.py`,
  `test_merge_status_answer.py`, `test_git_authority.py`, `_merge_tool_calls.py`; the
  merge_status/merge_cancel/merge_relane sections of `escalation/tests/test_server.py` and
  `test_server_chokepoint.py` → `orchestrator/tests/test_merge_tools.py` (new);
  `orchestrator/tests/test_merge_lane_cancel_request.py`, `orchestrator/tests/test_merge_status_boundary_b10.py`,
  `orchestrator/tests/test_citation_gate_positive_arm.py` (import), `scripts/tests/test_merge_state_vocabulary_consistency.py`
  (`PINNED_SITES` path), `orchestrator/tests/merge_lane_ratchet_baseline.json` (`--authorize-raise`).
- Keys: `fk-a8d3003e9d83` (part), `fk-e903cc072b15`, `fk-f8e9e0da3786`. Deletes: `_get_merge_journal`,
  `_map_live_state` and its pinned pass-through test, the `hasattr(worker, 'snapshot')` guards of the
  moved tools, 4 function-local + 2 `TYPE_CHECKING` imports, `DISALLOW_ESCALATION_MERGE_LANE_CONTROL`'s
  only member.
- Deps: ι (server.py), S6 μ (native port and `LIVE_STATE_OF`; held; `harness.py` after S6's chain,
  which follows κ), S8 ε2b (`harness.py`; lead ruling 2026-11-10), 4651 (lands first), 4833 (lands
  first; deletes the `retention=` keywords and `_get_terminal_retention`), S10 β1/β2
  (`_write_merge_lane_metadata`). Seams: C1 consumer; C7a; S6's `types.py`/`worker.py` edit named in
  the S6 xref; S8's `stranded_reconcile.py` import repoint.
- Signal: boundary row 3 (the tool half) and row 9 (merge read/cancel half); `merge_status` on a live
  request answers from the native port, on a landed one from its terminal tier;
  `escalation.git_authority` no longer importable.
- G7: INV-13 (the terminal tier is read from its producer, the journal and `merge_finalized` rows),
  INV-5 (one tier walk, one state mapping, one home), R8 (the moved filer goes through the ladder's
  one submit path).

**ε2 — `merge_request` from `orchestrator/mcp/merge_tools.py`; the dedupe gate on the native port; the
private probes go**
(`task_kind=normal`, priority high, est. ~650 LOC authored; ~800 tool lines moved)
- Files: `orchestrator/src/orchestrator/mcp/merge_tools.py`, `orchestrator/src/orchestrator/merge_lane/intake.py`
  (gate onto the port; predicates and `live_snapshot` deleted), `orchestrator/src/orchestrator/merge_lane/types.py`
  (`lane_unanswerable`), `escalation/src/escalation/server.py`, `escalation/src/escalation/server_instructions.py`,
  `orchestrator/src/orchestrator/harness.py`, `review/briefing.yaml`, `fused-memory/tests/test_stages.py`,
  `escalation/tests/test_merge_request_lane.py` → `orchestrator/tests/`, `orchestrator/tests/test_merge_queue.py`,
  `orchestrator/tests/test_merge_queue_c3_submit_identity.py` (dict fakes become port fakes),
  `orchestrator/tests/test_merge_lane_intake.py`, `orchestrator/tests/test_merge_tools.py`, the
  `merge_request` sections of `escalation/tests/test_server.py` and `test_server_chokepoint.py`,
  `orchestrator/tests/merge_lane_ratchet_baseline.json`.
- Keys: `fk-a8d3003e9d83` (part), `fk-e903cc072b15`, `fk-f8e9e0da3786`, `fk-c8c3406fd7f4` (the gate).
  Deletes: the three predicates (186 lines), the `live_snapshot` plumbing, seven silent fail-safe arms,
  `_get_merge_worker`, the lazy registry construction, the `merge_inflight_registry=`/`merge_waiters=`
  keywords, the function-local imports in `merge_request`, the briefing known-gap entry.
- Deps: ε1 (and through it 4833), S6 μ (held; `intake.py` after S6's lifecycle and limits tasks
  through it), 5409 (lands first), S7 δ (deletes `retry_failed_only` first). Seams: C7a complete.
- Signal: boundary rows 1, 2 and 9 (merge_request half); `git grep -E '^[[:space:]]*from orchestrator[. ]'
  escalation/src` is empty.
- G7: INV-11 (D3; the fail-safe is in the return value), INV-2, INV-13, INV-5 (one port, not five
  dict readers).

**ζ — ladder tools from three escalation modules; `LadderLookups`; `create_server` composition; layering guard**
(`task_kind=normal`, priority high, est. ~450 LOC authored; ~3,400 moved; ladder test call sites by substitution)
- Files: `escalation/src/escalation/filing_tools.py`, `resolve_tools.py`, `read_tools.py`, `lookups.py`
  (the last three new), `escalation/src/escalation/server.py`, `orchestrator/src/orchestrator/harness.py`
  (builds `LadderLookups`), `escalation/tests/_ladder_server.py` (new), `escalation/tests/test_layering.py` (new),
  `escalation/tests/test_create_server_shapes.py` (new; rows 8, 11), `escalation/tests/test_server.py`,
  `escalation/tests/test_server_chokepoint.py`, `escalation/tests/test_pins_recovery_annotation.py`,
  `escalation/tests/test_missing_argument_registration.py`, `escalation/tests/test_note_to_file.py`,
  `escalation/tests/test_resolve_escalation_shas_tool.py`, `escalation/tests/test_pending_scan_off_loop.py`,
  `escalation/tests/test_server_eval_lane.py`, `orchestrator/tests/test_repend_state_machine.py`,
  `orchestrator/tests/_escalation_server_factory.py`, `scripts/tests/test_sitting_payloads.py`.
- Keys: `fk-a8d3003e9d83`, `fk-e903cc072b15` (the guard). Deletes: `harness: Any` and the
  `merge_queue`, `orch_config`, `event_store`, `task_status_lookup`, `task_claimant_lookup` parameters
  from `create_server`; `getattr(harness, 'pre_increment_routing_tier'|'granted_files_delivery_blocker'|
  'scheduler'|'git_ops')`.
- Deps: ε2 (held through it). Seams: C3 (`workflow_active` bound to `WorkflowSlots.is_active`), C7b
  (fused-memory harness unchanged, `prefix_resolver` kept); S8 ε3 and S8 ω depend on ζ (ω then deletes
  the `Harness.is_workflow_active` forwarder).
- Signal: boundary rows 8, 10, 11; the fused-memory reconciliation server on :8103 lists 13 tools after
  a fused-memory restart; `escalation/tests/test_layering.py` fails on a seeded fixture module and
  passes on the tree.
- G7: INV-10 (the guard is exercised against a fixture module), INV-1 (lookups are a declared bundle).

**λ — tool descriptions and module prose under the Comments stance**
(`task_kind=normal`, priority medium, est. ~400 LOC net; large character delta)
- Files: `escalation/src/escalation/filing_tools.py`, `resolve_tools.py`, `read_tools.py`, `queue.py`,
  `orchestrator/src/orchestrator/mcp/merge_tools.py`, `ops_tools.py`.
- Keys: `fk-43f431f697a3`. Deletes: maintainer/reviewer prose from the 29 tool descriptions (the
  measured 99,454 chars and 0.58 share are the baseline; the leaf records the after figures in its
  commit, not a target).
- Deps: ζ (and through it every other leaf and 5409). Seams: none.
- Signal: `mcp__escalation__*` tool descriptions listed by a client contain no `task \d+`, no `please
  do not`, no `reviewer`; the before/after imperative lists in the commit are equal.
- G7: INV-9.

**μ — integration gate and companion docs** (`task_kind=normal`, priority high, est. ~450 LOC)
- Files: `escalation/src/escalation/testing_http.py` (new; lifted from `escalation/tests/conftest.py`),
  `escalation/tests/conftest.py`, `orchestrator/tests/conftest.py`, `escalation/tests/test_s3_boundary.py` (new),
  `orchestrator/tests/test_s3_boundary.py` (new), `ARCHITECTURE.md`,
  `skills/merge-queue/references/two-layer-model.md`, `OPERATIONS.md` (tool-list mentions if any).
- Keys: `fk-a8d3003e9d83` (gate). Deletes: nothing.
- Deps: λ (transitively all). Signal: rows 1–17 green over real HTTP, each in its suite; after the
  dark-factory orchestrator restarts on the new code (fleet redeploy) AND fused-memory restarts on it —
  through `scripts/restart-fused-memory.sh` or the fused-memory staleness backstop in
  `scripts/orchestrator-watchdog.py`, since the fleet sweep never restarts fused-memory — the live
  :8102 server lists 29 tools, :8103 lists 13, and pass 0 has indexed every archived record.

DAG ("→" = lands before): S6 α → β → γ → θ; 4567 → θ; α → δ → κ → θ → ι → ε1 → ε2 → ζ → λ → μ;
β → δ; γ → κ; κ → S6 β1 (… S6 μ); α → S6 μ → ε1, ε2; S8 ε2b → ε1; 4651 → ε1; 4833 → ε1;
S10 β1/β2 → ε1; 5409 → ε2; S7 δ → ε2; β, δ → S8 β; δ → S8 α; ζ → S8 ε3, S8 ω; δ → S5 β. S6 μ has no
edge from any S3 merge-tool move; no S8 leaf at or before ε2b depends on an S3 leaf after it; S6 α
depends on no S3 leaf and takes no edge on S5 α, S5 λ0, S7 δ or S7 β; S7 δ depends on no S3 leaf at
or after ε2 (each such edge closes a cycle or holds S6 α). Held on S6 μ: ε1, ε2, and through them
ζ, λ, μ.

## Live tasks (contract §8 step 2)

Statuses re-read with `get_task`/`get_statuses` on 2026-11-10.

- **6319** done (`0935e63d88`). **6670** done (`f6a4482199`): it added `runner` to the workflow-side
  green; nothing to wire.
- **5409** (pending, medium): owns `fk-bc64099a1cd0`; amend (key, constructor contract, `runner`) and
  wire `add_dependency(ε2, 5409)` — 5409 lands first in today's files (OQ-1 ruling), so its declared
  files keep `escalation/src/escalation/server.py` and `escalation/tests/test_server.py`.
- **4651** (pending, high; waits on 4646, on S8 ε2a by S8-2, and temporarily on 5040 through the
  lead's parking edge, which S8's decompose removes): `add_dependency(ε1, 4651)` — it lands first
  against `escalation.git_authority`, inside S8's `stranded_reconcile.py`, and ε1 repoints the import.
- **4567** (pending, high; 8 commits on `task/4567` dated 2026-11-07, no live claimant):
  `add_dependency(θ, 4567)` (`queue.py`; θ only, not β — lead ruling 2026-11-10); it moves the
  in-process archive-inclusive calls off the loop, and θ then removes most of their cost.
- **4833** (pending, low; dispatchable today; carries `fk-66746efddd4d`): lands FIRST, in today's
  files (lead ruling 2026-11-10, reversing the earlier 4833 → ε2): `add_dependency(ε1, 4833)`; S6 κ → 4833 is S6's edge, and 4833 waits on
  no S3 or S6 task. Amend:
  `x_finding_key` gains `fk-66746efddd4d` (`x_finding_run: 'review-all-dark_factory-20261008'`);
  `metadata.files` swaps `merge_types.py`/`merge_queue.py` for today's `merge_lane/types.py`,
  `merge_lane/intake.py`, `merge_lane/gates.py`, `merge_lane/worker.py`, `merge_lane/queue_store.py`
  and keeps `escalation/src/escalation/server.py`; its `tier2-read-already-removed` check stays on
  `server.py` (it lands before the move).
- **6571** (pending): cancel at decompose, `x_acceptance_reason = "subsumed by S3 ε1: the tier walk
  moves to orchestrator/mcp/merge_tools.py under R10"`; if claimed, ε1 depends on it instead.
- **6827** (pending, low): cancel at decompose as covered — `x_acceptance_reason = "covered: the
  merge-lane plan's native-port task (S6 μ) maps every non-terminal ItemLifecycleState to MergeState
  through one total, test-checked table (REDISPATCH_PARKED = wire 'awaiting_host', DISPATCHING), the
  'passthrough' phase no longer exists, and S3 ε1 deletes _map_live_state and its pass-through"`; if
  claimed, ε1 depends on it instead.
- **6004** (pending, low): θ removes its premise. **3176** (pending): same-file edge with κ on
  `harness.py`. **6920** (pending, low): related to ι's gate table and β's locked mutation; untouched.
  **3149**, **2932** (pending): closed by 4833.
- **4552** (deferred) predates task 4885 (done), which put `patch_resolution_metadata` under the
  per-id lock; β's primitive keeps that property. Not a dependency.

## Out of scope

- `_chokepoint_or_submit`'s five gates and the filing entry point (S1, R8).
- `Escalation` as a typed/frozen record, `cascade_parent_id` as a field, `parse_caller_identity` (S2, R9).
- Decomposing `merge_request` itself (820 lines): it moves whole; its h4 cost is a lane concern
  recorded for S6/S7.
- `SpeculativeMergeWorker.snapshot()` as a rendering; the native `MergeLaneQueries` implementation and
  the `LIVE_STATE_OF` table (S6 μ).
- The `workflow_verify` constructor (task 5409).
- `make_id`'s per-key archive scan in `_recover_seq_from_disk` (no key of this stream).
- Moving queue I/O off the event loop in `async def` tools (task 4567).
- fused-memory's reconciliation `merge_coalesced_payload` (a different domain's "coalesce").

## Rulings (sitting of 2026-11-10)

Every design question of this plan is ruled; the text above is authored under these rulings. None of
them amends Leo's 2026-10-10 ruling R10.

| Id | Ruling | By |
|---|---|---|
| OQ-1 | Option (a): the merge tools wait for S6's native lane-query task; no interim adapter. α shrinks to the Protocol, records, error and rename; ε2 takes the gate conversion, the predicate deletion and `lane_unanswerable`; ε1/ε2 depend on S6 μ, S6 μ does not depend on ε2; `harness.py` order interleaved; tasks 5409 and 4651 land first in today's files; θ, ι, κ placed before the ladder split; S6 μ's priority comes from inheritance, not a hand raise. | Leo |
| OQ-2 | As recommended, (a): all three emitters write `merge_deduplicated`; `merge_coalesced` stays read-only legacy; the closed rename set; two stored values keep their spelling. | Leo |
| OQ-3 | Three family modules (`filing_tools.py`, `resolve_tools.py`, `read_tools.py`), each with its own registrar. | the lead |
| OQ-4 | Names and placement as listed; `git_authority.py` under `orchestrator/mcp/`, `lane_resolution.py` in `merge_lane/`, `queries.py` its own module. | the lead |
| OQ-5 | As recommended, (a): `deliver_on` loop marshalling in the queue, closed-loop drop counted, the timing change stated. | Leo |
| OQ-6 | Option (a): index the archive tier only; the pending tier stays the root glob. The plan's sentence on fused-memory writers is corrected (two long-running writers per project directory, restarting on separate clocks), and the integration signal names a fused-memory restart through `scripts/restart-fused-memory.sh` or its staleness backstop. | Leo |
| OQ-7 | (a): S3 C2 declares `WorkflowSlots(is_active, cancel, wait_cleared)`; `Harness` implements it until S8's `SlotRegistry` does; S8 drops its own definition. | the lead |
| OQ-8 | Superseded by Leo: one internal enum (`ItemLifecycleState`) and one total, test-checked mapping to the public `MergeState` inside the orchestrator; no raw state crosses the port; the internal enum is not made public; task 6827 cancelled as covered. | Leo |

Lead sequencing rulings after the ruling folds (2026-11-10, ruled by the lead; they win over earlier
text), as applied above:

| Ruling | Effect on this plan |
|---|---|
| `harness.py` order: S6 α first, then S3 β, δ, κ, then S6 β1 … μ, then S3 ε1, ε2, ζ; S6 α never held behind S3 | β depends on S6 α; S6 β1 depends on κ; κ placed first of κ, θ, ι on `server.py` so S6's chain never waits on 4567 |
| Task 4833: edge reversed — 4833 lands first in today's files | ε1 depends on 4833; `fk-66746efddd4d` → `filed:4833`; ε1/ε2 no longer delete `retention=` or `_get_terminal_retention` |
| Task 4567 is a prerequisite of θ only, not β | θ depends on 4567; β does not |
| S8 ↔ S3 cycle cut confirmed | S8 β depends only on S3 β and δ; S3 ε1 depends on S8 ε2b; S8 ε3 and S8 ω depend on S3 ζ; 4651's `git_authority` import lives in `stranded_reconcile.py`, where ε1 repoints it |
| S7 δ first in S7's chain; S3 ε2 depends on S7 δ | unchanged edge; S7 δ must not depend on ε2 or a later S3 leaf |
| S6 α lands first on every file it shares with another plan; it takes no edge on S5 α, S5 λ0, S7 δ or S7 β (alignment pass, 2026-11-10) | β … ι wait on S6 α alone, on no S5 or S7 task; S7 δ against α … ι on `server.py` is the lead's per-file order |
| Task 4833 lands first in today's files; S6 κ → 4833 (alignment pass, 2026-11-10) | 4833 waits on no S3 or S6 task; ε1 depends on 4833; α's order against 4833 on their shared files is the lead's per-file order, never 4833 after α |

## Open questions

None.

Tactical (decide in the leaf): the exact `QueryUnavailable` event name (ε2); whether `MutationResult`
also carries the pre-mutation record for `amend`'s `repeat_framing` arm (β); lock sidecar placement for
`by-task/` appends and the trailing-partial-line tolerance (θ); whether `MERGE_TOOLS_INSTRUCTIONS` is
appended or the server instructions become a registrar-contributed list (ε2); the `MergeWaiters` type's
shape (ε1); the home of each tool's moved rationale (λ).
