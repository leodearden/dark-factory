# Held-follow-up packing sweep: a periodic curator arm that packs coherent held follow-ups into one survivor, observe first

**Status:** active PRD, authored 2026-11-04 (`/prd` author seat under `/team`, from Leo's rulings of
2026-11-03/04 on `plans/portfolio-review-2026-11-01/followups/task-shape.md` option B — observe
mode first, no age-out, an accepted pack promotes its survivor — and the lead's design decisions
recorded in §3; critic pass `task-shape-seats/PRD2-C.md` folded in, §11). B+H: three packages
touched (`fused-memory`, `shared`, `scripts`), nine mechanisms, two load-bearing seams (the
interceptor's status chokepoint and combine write; PRD 1's hold and promotion records). Sibling of
PRD 1 (`plans/follow-up-filing-shape-prd.md`, tasks 8413–8421) and of the riders PRD (option C, to
author).
**Code anchors** cite symbols; verified against main `8d45e1edca` (2026-11-04). Main moves fast;
re-locate at implementation time. Symbols that PRD 1 introduces (`followup_hold.py`,
`followup_items.py`, `declared_scope.py`, `promote`, `merge_items`) are cited as PRD 1 §4 contract
names and are NOT on main yet (§7).
**Evidence:** `followups/task-shape.md` §5 (S5 combinability) and §7 option B; design scouts
`task-shape-seats/PRD2-M.md` (execution: ledger, loop, executor, undo, claim race, the 5197
precedent) and `task-shape-seats/PRD2-P.md` (candidate generation, judge contract, hub rule,
cadence, reading check); the a2 + c2 re-run of 2026-11-04 over today's stores: its input, the full
held member set, is `plans/followup-packing-sweep-prd.pack-pool-2026-11-04.json` (2,517 records,
df 1,405 + reify 1,112, with the rule, read time, lock depth and hub keys in its header) and its
output is `plans/followup-packing-sweep-prd.packs-a2c2.csv` (both projects, whole pool and minus
the rider reservation, hub rule share > 2%). Those two files are committed with this PRD and are
the source of every yield, coverage, judge-call, cost and drain figure below.

## 1. Goal

Once a day, per project, a curator arm reads the held follow-up pool PRD 1 creates, proposes small
packs of follow-ups that are one coherent task (same parent, same kind and a shared non-hub lock
key, or the same mechanical recipe), has the curator model judge each pack's coherence, and records
every proposal and verdict in its own ledger. For at least a week it only records. After Leo reads
the report and flips the mode, an accepted pack is executed: the oldest member is rewritten in
place as the survivor carrying the union of items, the other members are cancelled with
`superseded_by` pointing at it, and the survivor is promoted out of the hold so the pack lands as
one landing instead of N. The measured premise (re-run of 2026-11-04, rules in §2 last bullet): on
today's whole held pool the a2 + c2 generators under the > 2% hub rule cover df 339 of 1,405 held
tasks in 118 packs (221 landings avoided before the judge, 24% of the pool) and reify 385 of 1,112
in 134 (251, 35%); after the judge's measured reject rates (0.6 kept on a2, 0.8 on c2) about df 158
and reify 177 landings avoided (11% / 16% of the held pool), at df 19–56 and reify 86–110 host
verify minutes per avoided landing (task-shape §7). The rider reservation (decision 13) defers
about half of df's with-files members from the first observe week; the figures the observe report
will actually show are in decision 1.

### What an operator observes when this lands

- The fused-memory start log prints `Pack sweep: mode=observe interval=86400s ...` beside the
  ticket-janitor line; `$DARK_FACTORY_DATA_DIR/pack_sweep.db` exists and its `runs` table gains one
  row per project per day with `judged`, `proposed`, `rejected`, `stopped_reason`.
- `scripts/pack_sweep_report.py` prints, per project: proposals and verdicts by generator, reject
  and split rates, pack size and union-key p50 / p90, judge calls and dollars, coverage of the held
  pool as two series (the reservation-eligible pool and the whole held pool), exclusion counts by
  reason, recipe hit counts, and a seeded sample of proposals rendered as member titles plus the
  survivor rewrite; a project whose ledger has no `sqlite_sequence` row prints `producer has not
  written`, never `0`.
- In observe mode `tasks.db` is byte-identical before and after a tick, and a second tick over an
  unchanged pool makes zero judge calls.
- Seven days after the observe leaf and the report leaf land, one L2 (a deterministic
  always-escalates gate) asks Leo to read the report and set `curator.pack_sweep.mode: enforce`
  through `reload_config`; `mode: off` is the kill switch at any time, no restart.
- In enforce mode, `get_task(survivor)` shows `metadata.pack_record{proposal_id, generator,
  members, packed_at}`, `followup_items` equal to the union of the members' items,
  `declared_scope.basis == 'pack'`, no holder edge, and a `followup_promoted{by: 'pack', ref:
  'pack:<proposal_id>'}` event; each other member shows `status: cancelled` and
  `metadata.superseded_by: [survivor]`; no L1 and no residue record is filed for the pack.
- A member or survivor the scheduler claimed, or that was promoted or edited, between judgement
  and execution is never rewritten or cancelled: the proposal records `refused` / `stale` (or
  `partial` with the executed steps) and the operator sees it in the report.
- `mcp__fused-memory__undo_pack(proposal_id)` reopens the members `pending` with
  `reopen_reason='pack-undo:<proposal_id>'` (each with a fresh `pending_since`, by PRD D3),
  re-holds them, restores the survivor's text and items from its pre-image and re-holds it; the
  ledger row reads `undone`.

### Consumers (G1)

| mechanism | consumer |
|---|---|
| candidate generators a2 and c2, hub rule, exclusions incl. `rider_reserved`, survivor rule (`followup_packing.py`, P0) | the sweep tick (P1), the only caller; the report (P3) reads generator, recipe and exclusion names from the ledger; the riders PRD cites `RIDER_RESERVE_DAYS` and `rider_reserved` (§5) |
| pack judge: `CURATOR_PACK_SPEC`, `_PACK_CONTRACT`, `CURATOR_PACK_OUTPUT_SCHEMA`, verdict parser, `render_pack_member`, `judge_pack` (`pack_judge.py`, P0) | the sweep tick (P1); the prompt-artifact store (`PromptArtifactStore.resolve`) under its own harness version, so a future pin keyed `(curator_pack, model, curator-pack-v1)` is clean of the filing curator's artifacts |
| ledger `pack_sweep.db` (runs / proposals / outcomes, P1) | the report (P3); the executor (P2) reads `proposed` rows and writes outcomes; the flip gate (Φ) reads the report; the steward reads `pack_record` on a blocked survivor (existing ladder) |
| `PackSweep` loop and `PackSweepConfig` with every leaf in `RELOADABLE_FIELDS` (P1) | `server/main.py` starts it beside the ticket janitor; the operator through `fused-memory/config/config.yaml` and `reload_config`; the `runs` rows and the start-log line |
| `_write_combined_rewrite` extracted from `_execute_combine` (P2) | `_execute_combine` (filing-time combine, lock-free as today) and `apply_pack` (under the write lock after a re-read); PRD 1 α's `DeclaredScope.basis` gains `'pack'` and its authority rule keeps one writer site |
| `_apply_status_transition(require_before=...)` precondition and its thin public wrapper `cancel_superseded_if_pending` (P2) | `apply_pack`; the cancel-with-children PRD §4.14 entry-point table gains a "pack executor" row (companion correction) |
| `apply_pack` and `undo_pack` (P2), `undo_pack` MCP tool | the enforce tick; the operator and `/unblock` on a blocked survivor; the integration gate (P3) |
| `followup_hold.rehold` (P2, the inverse of `promote`) | `undo_pack` only |
| report (P3) and flip gate (Φ) | the operator; Φ is the INV-13 proof that the producers above wrote in production |

## 2. Background (substrate as it is)

- Periodic in-process sweeps in fused-memory have one loop shape:
  `fused-memory/src/fused_memory/middleware/ticket_janitor.py::TicketJanitor.run_loop` (sleep, then
  `tick()`, every step wrapped, cancellation the only exit), built by `server/main.py::_build_ticket_janitor`
  over `build_known_projects_map` and started with
  `asyncio.create_task(ticket_janitor.run_loop(janitor_cfg.interval_seconds))` — the interval is
  captured by value at start, so in that shape it is NOT hot; cancelled in shutdown. Config
  `CuratorConfig.janitor: TicketJanitorConfig` in `fused-memory/src/fused_memory/config/schema.py`,
  values in `fused-memory/config/config.yaml` under `curator.janitor`. The link-healing executor
  (`fused-memory/src/fused_memory/maintenance/link_heal_executor.py`, ledger `link_heal_ledger.py`,
  `LinkHealConfig{max_actions_per_run=25, write_failure_streak=3, ...}`) is the plan → apply → undo
  discipline: its own sqlite with `AUTOINCREMENT` ids (a `sqlite_sequence` row proves a producer
  wrote, INV-13), a plan run that writes nothing to the store, apply that re-reads each record
  live and skips if changed, a per-run cap, a failure streak that stops and escalates through
  `_folded_escalation.file_folded_escalation`, and an undo that takes an apply run back
  newest-first and suppresses re-proposal at the same texts (PRD2-M §1.1).
- Nothing on main combines two pending tasks. `task_interceptor.py::TaskInterceptor._execute_combine`
  folds one ticket candidate into one pending target and takes NO `_write_lock`: its guards (a live
  `get_task`, a `normalize_title` fingerprint, `curator_types.py::is_combine_eligible_status`, a
  present `claimant_run_id` stamp (`COMBINE_REFUSED_CLAIMED`), `_has_branch_base_sha`
  (`COMBINE_REFUSED_HAS_BRANCH`), a gated-candidate rule) and its `tm.update_task` are a lock-free
  check-then-write against the backend. The write is a `FILES_TO_MODIFY:` details block built from
  the rewrite, `metadata_mode='merge'` of `{curator_action, curator_justification, combined_at}`,
  an audit row through `_append_combine_audit` (path `_combine_audit_path`, under
  `DARK_FACTORY_DATA_DIR`, append-only, description truncated to 500 chars) and
  `_advise_text_amendment(writer_source=AmendmentWriter.CURATOR_COMBINE)`. PRD 1 γ (8415) adds the
  item merge and the `declared_scope` re-stamp (`basis='combine'`) into that same write region
  (PRD2-M §1.2; PRD2-C AC2).
- The interceptor's public `update_task` holds `_write_lock(project_id)` around its read and
  write; `_apply_status_transition` acquires the same lock itself and reads `before` inside it (the
  CSV branch of `set_task_status` calls it directly, building the `DeferralCall` it needs).
  `_write_lock` is a per-project, non-reentrant `asyncio.Lock`: a wrapper that takes the lock and
  then awaits `_apply_status_transition` deadlocks on itself. `TaskInterceptor.set_task_status`
  has `reopen_reason` and `expected_stamped_at` but no `expected_status`.
- Claims: `orchestrator/src/orchestrator/scheduler.py::Scheduler.set_task_status` writes
  `in-progress` with `claimant_run_id` / `heartbeat_at` through the MCP `set_task_status`, which
  lands in `_apply_status_transition` under that lock. A precondition evaluated against `before`
  inside that critical section is therefore a compare-and-set against the claim write; a
  check-then-cancel outside it can cancel a task claimed in between (PRD2-M §1.5, PRD2-C AC1).
- The curator's prompts are `PromptSpec(prompt_id, contract, baseline_heuristics)` objects
  (`shared/src/shared/prompt_artifact.py::PromptSpec`, composed by `compose_prompt`):
  `task_curator.py::CURATOR_SINGLE_SPEC` and `CURATOR_BATCH_SPEC` share `_CURATOR_HEURISTICS` by
  reference and differ only in contract (`_CURATOR_CONTRACT`, `_BATCH_CONTRACT`); they resolve
  through `PromptArtifactStore.resolve(spec, executor_model, harness_version)` with
  `_CURATOR_PROMPT_HARNESS_VERSION = 'curator-v1'`; a pinned artifact is keyed by the
  `(prompt_id, executor_model, harness_version)` triple (also read by
  `orchestrator/src/orchestrator/evals/prompt_opt/canary.py`). `_CURATOR_HEURISTICS` rules 1–4
  are coherence ("no 'and also'"), narrow lock (union within the target's 1–2 lock keys), planning
  budget (≤ 15 files, ≤ 5 concerns) and no critical-path drag. `CURATOR_BATCH_OUTPUT_SCHEMA` has
  no pack-level verdict and no split; `_parse_decision_dict` degrades an invalid target to create,
  audited. `PoolEntry.render` clips through `clip_for_prompt`; `PoolEntry.source` is one of
  `POOL_SOURCES` and `combine_eligible` is a filing-pool concept, so `PoolEntry` is not the pack
  render shape (PRD2-P §4). `TaskCurator._ledger_kwargs(role: str)` takes any role string.
  `task_curator.py` is 3,951 lines and task 6707 (pending high) restructures it.
- LLM substrate: `CuratorConfig{model='sonnet', max_budget_usd=2.00, batch_max=5,
  batch_budget_cap_usd=2.00, zero_output_breaker_threshold=2}`; calls go through
  `invoke_with_cap_retry`; the stop vocabulary is `middleware/curation_deferral.py::DeferReason`
  (`AllAccountsCappedException`, `PoolFrozen`, the ZOT breaker). The same module owns
  `DEDUPE_SKIPPED_KEY = 'dedupe_skipped'` and `dedupe_skipped_marker`, the stamp task 6322's sweep
  will re-dedupe (PRD2-M §1.6, §8). The measured full-pool curator call is $0.306.
- Lock keys: `shared/src/shared/locking.py::files_to_modules(files, depth)` after
  `strip_directory_locks`; `normalize_lock` is the identity on every declared file in both df
  (`lock_depth: 12`) and reify (`lock_depth: 10`) stores today (none deeper than 10 components).
  The curator resolves the depth per project through
  `fused-memory/src/fused_memory/mcp_tools/scheduler_state.py::effective_lock_depth(project_root,
  config.curator.lock_depth)`, imported into `task_curator.py` (PRD2-P §5; PRD2-C b-1).
- Every `cancelled` transition fires targeted reconciliation;
  `reconciliation/targeted.py::TargetedReconciler._sweep_cancelled_descendants` scans every task
  whose `spawned_from == id` or whose `dependencies` contain `id`, files one L1 per live descendant
  and (after cancel-with-children γ 8144) skips children named in the parent's `superseded_by`. A
  held follow-up's only dependency is the holder, so a member cancel files nothing iff the member
  has no live dependent and no live `spawned_from` child; each cancel is still one reconciliation
  run (5197's 164 transitions restarted fused-memory once; PRD2-M §1.4, §2). `superseded_by` is not
  yet in `shared/src/shared/task_metadata.py::_BLESSED_METADATA_KEYS` (the hits on main are the
  memory-store key); cancel-with-children α 8142 blesses it and β 8143 writes it atomically with
  the cancel through `sqlite_task_backend.py::set_status_and_stamp_audit`, editing
  `_apply_status_transition` and `set_task_status`; both are `pending` (store read 2026-11-04).
  `docs/task-authoring.md` §2 lists `done/cancelled ─reopen (audit, requires reopen_reason)→ *`;
  `sqlite_task_backend.py::stamp_pending_since` overwrites `pending_since` with `now` on
  `cancelled → pending` (D3), and `scheduler.py::_update_age_anchors` resets the dispatch age on
  that stamp.
- The starvation watchdog ages a task from `eligible_since`
  (`orchestrator/src/orchestrator/eligibility_clock.py::age_secs`, anchored the first tick the task
  is a dispatch candidate; a task with no anchor reads 0.0) and never reads `pending_since`. A held
  task is never a candidate, so a promoted survivor's watchdog clock starts at promotion. What the
  survivor keeps by keeping its id is the dispatch-scoring age (`_update_age_anchors`, anchored on
  id order), which is what helps a low survivor dispatch (PRD2-M lesson 2c; PRD2-C AC4).
- Precedent 5197 (done): a hand-run sweep that coalesced 08-28 170 members → 60 carriers and 09-09
  164 members → 64 carriers (5254–5317, `source='backlog-sweep-coalesce'`), members parked
  `deferred` with `x_coalesced_into`. Store today: 91 members still deferred behind pending
  carriers, 21 deferred behind done carriers (residue nothing closes), 48 cancelled by hand, 34 of
  64 and 21 of 60 carriers still pending after 8–10 weeks; the planning-mode `submit_task` deduped
  one merged spec onto a member through `candidate_key` (5011); carrier 5292 inherited a
  dependency one member did not need (5089, Leo ruling esc-5580-4); the ledger lived in `/tmp`
  (PRD2-M §2).
- Pool and yields, re-run 2026-11-04 over both stores (`?mode=ro`, tag `master`, `status='pending'`,
  PRD 1 decision 9's hold rule; input `followup-packing-sweep-prd.pack-pool-2026-11-04.json`,
  output `followup-packing-sweep-prd.packs-a2c2.csv`, both beside this PRD). Rules exactly as
  PRD2-P §0 states them, c2's ordering pinned in the dump header and §4.1, with none of decision
  7's exclusions and no decision 8 partition applied (PRD2-P applied neither; the dump carries
  `has_live_dependents` and `band_proxy` per member so the report's exclusion counts can be
  previewed, 144 members have a live dependent or child). The cells are therefore upper bounds on
  the enforce regime; decision 1 quantifies the gap:
  a2 = same `spawned_from` and same kind, members limited to the union-find components inside the
  parent group that share a non-hub file, chunked greedily in id order under ≤ 5 members and ≤ 15
  union files; c2 = the same tightened recipe (`h14_relocate`, `cite_doc_hygiene` v1,
  `spot_dedupe`, `test_seam`, `diag_code`), members sorted by the directory of their first file,
  chunked under the same budget; precedence a2 > c2, disjoint; hubs = files declared by > 2% of the
  pool (df 8, reify 16). Cells are packs / tasks / landings avoided (Σ |pack| − 1).

  | pool | df (held 1,405; 1,023 with files) | reify (held 1,112; 1,012 with files) |
  |---|---|---|
  | whole pool: a2 | 77 / 171 / 94 | 89 / 206 / 117 |
  | whole pool: c2 | 41 / 168 / 127 (spot 24/98/74, h14 7/28/21, cite 4/20/16, seam 6/22/16) | 45 / 179 / 134 (cite 22/79/57, spot 13/57/44, h14 6/27/21, diag 2/10/8, seam 2/6/4) |
  | whole pool: union | 118 / 339 / 221; 24.1% of held covered; after judge ≈ 158 (11.2%) | 134 / 385 / 251; 34.6%; ≈ 177 (16.0%) |
  | minus rider reservation (files and `pending_since` < 14 d: df 467, reify 169 excluded) | pool 938; a2 34 / 74 / 40, c2 30 / 124 / 94, union 64 / 198 / 134; 14.1% of held (21.1% of the pool); after judge ≈ 99 (7.1%) | pool 943; a2 76 / 176 / 100, c2 36 / 147 / 111, union 112 / 323 / 211; 29.0% of held (34.3%); ≈ 149 (13.4%) |

  Under PRD2-P's top-5 hub rule the whole-pool unions are df 125 / 353 / 228 and reify 153 / 428 /
  275; the hub rule moves the union by ≤ 6% in df and ≤ 12% in reify. Reviewer suggestions carry
  `spawned_from` but no `files` (`review_suggestions/curator_intake.py` files none), so they reach
  packs only through c2. Reading check (PRD2-P §2, n = 5 per generator): the judge must reject or
  split about 40% of same-parent packs and about 20% of recipe packs; loose same-parent would be
  80%; the title-verb generator matched unrelated asks on a shared verb in 2 of 5. Pool
  `pending_since` runs df 07-23 → 11-04 (median 10-25), reify 07-25 → 11-04 (median 10-04).

## 3. Resolved design decisions

Leo, 2026-11-03/04: the sweep runs in **observe mode first** (proposes and records for a week before
any rewrite); **no age-out** for held tasks; **an accepted pack promotes its survivor**
(`promote(by='pack')`, PRD 1 §4.3), so packs land.

Lead's decisions under those rulings:

1. **Boundedness is stated, not claimed, from the re-run.** Over the whole held pool, a2 + c2 cover
   24% (df) and 35% (reify) before the judge and about 11% / 16% after it (§2 table). The pool the
   first observe week actually sees is the reservation-eligible one (decision 13): df 938 of 1,405
   held (the reservation defers 467, nearly half of df's 1,023 with-files members and 54 of its 118
   packs), reify 943 of 1,112; coverage there is df 14% of held before the judge and ≈ 7% after,
   reify 29% / ≈ 13%. The 118 / 134 packs are upper bounds taken before decision 7's exclusions
   and decision 8's dependency partition: against the live stores, 9 of 118 df and 25 of 134 reify
   whole-pool packs contain members with unequal non-holder dependency sets (e.g. 6905 behind 8032
   beside 7007 behind nothing; 6554 behind Ω 9739 beside 6576 behind 5470 + 9739), and the
   live-dependents exclusion alone moves a2 77 → 74 (df) and 89 → 83 (reify), c2 41 → 39 and
   45 → 44; some of those packs re-form smaller rather than vanish, so the regime figure is below
   these by an unmeasured amount of up to ~19% of reify's packs, and the observe week measures it.
   The held pool is NOT bounded by packing alone: 65–90% of held inflow has no
   pack and stays held until a citation, dependent, rider or human promotes it. PRD 1 θ's 14-day
   report (8421) is where the age-out ruling is reopened with numbers; this PRD does not reopen it
   (§9 item 1). Φ's L2 quotes the a2 + c2 figures, both series.
2. **Members are closed `cancelled` with `superseded_by=[survivor]`** through the
   cancel-with-children vocabulary (that PRD §4.8: `superseded_by` on the cancelled task is the
   canonical supersession key; `fold{into}` is a child disposition and a member is nobody's child
   here). Never `deferred` (the 5197 residue) and never a legacy `x_coalesced_into`. So the enforce
   leaf depends on 8142 (vocabulary) and 8143 (atomic stamp), both pending; observe mode needs
   neither.
3. **The survivor is a member rewritten in place, never a new carrier.** Survivor = the member
   with the earliest `pending_since`, lowest id on a tie, derived server-side. In-place rewrite
   avoids the 5197 `candidate_key` collision (5011) and keeps the id, holder edge, `pending_since`,
   dispatch-scoring age and history. Survivor priority is its own, never raised by a pack (parity
   with the hold rule); P3's report counts survivor dispatch and landing within 30 days so the
   question can be reopened with numbers (PRD2-M §2 lesson c: 34 of 64 carriers never dispatched).
4. **Generators v1 are a2 and c2 only** (rules in §2). The title-verb generator b is DROPPED from
   v1 (lowest yield and noisy: a verb from free titles was wrong in 2 of 5 readings) and recorded as
   a v2 candidate (§8). Precedence a2 > c2; the partition is disjoint; a task is in at most one
   pack.
5. **Hub rule: a lock key declared by more than 2% of the pool, computed from the pool at sweep
   time** (`hub_keys(members, share=0.02)`), on lock keys produced by `shared.locking` at the
   project's `effective_lock_depth` (reused from `mcp_tools/scheduler_state.py`; SPOT with the lock
   table; directory charters dropped by `strip_directory_locks`). Never read from
   `module_overrides` (lock-width policy, a different fact) and never a top-k (does not scale from a
   37-task pool to a 2,000-task one).
6. **Bounds: Σ `item_count` ≤ `MAX_FOLLOWUP_ITEMS` (5) and |union lock keys| ≤
   `PACK_MAX_UNION_KEYS` (15).** The 5 keeps its home in `shared.followup_items`; the 15 gets its
   code home in `followup_packing.PACK_MAX_UNION_KEYS`, and the "≤ 15 files" prose in
   `_CURATOR_HEURISTICS` becomes a derived restatement (the heuristics text is not edited: a pinned
   artifact stays valid). `PackSweepConfig.max_pack_items` and `max_union_keys` default to those
   constants and are validated `≤` them: config may tighten a bound during observe, never loosen
   it, so the ceiling has one home.
7. **Exclusions, in the pure generator** (`exclusion(member, now)`): the holder; not held or already
   promoted; `declared_scope.band == 'broad'` (an all-narrow gate removes 80–95% of yield, PRD2-P
   §1b; `broad` is the defensible lever); any live dependent or live `spawned_from` child;
   `plan_authored_at`, `branch_base_sha` or a `claimant_run_id` stamp (a plan or branch would be
   invalidated by the rewrite); `metadata.dedupe_skipped` (the 6322 seam: packing a possible
   duplicate hides it); a carried rider — `RIDER` fires on `rider_of` set, exactly as PRD 3
   (`plans/followup-riders-prd.md`, decisions 8, 10, 11, 18) defines it: a carried rider, never
   left set on a rider whose host is terminal or stalled, because PRD 3's close / release clears
   it inside the host's own `done` or `cancelled` transaction and its 14-day carry sweep clears it
   on a stalled host; and a rider-reserved task (decision 13). `exclusion()` also reads
   `rider_ledger.last_decline_reason` when present as a same-parent pack HINT, not an exclusion: a
   `different_intent` decline means the task is packed with same-parent (a2) or same-recipe (c2)
   siblings, never offered into that host's area again by the packer; an absent key or field
   reads as no hint. A member whose `declared_scope` is absent (pre-deploy stock) is band-unknown and eligible
   (PRD 1 §4.2: absent means no information).
8. **Unequal non-holder dependency sets never pack together** (the 5089 lesson). The generator
   partitions each group by its non-holder dependency set before chunking, so a reify follow-up
   behind Ω 9739 packs only with others behind Ω; the executor re-checks and refuses
   `dependency_mismatch` if a set changed since judgement. No union of dependency sets, ever. A
   corollary: no in-pack dependency edge can exist (A depending on B puts B in A's set and not in
   B's), so cancellation order is plain ascending id.
9. **The judge lives in a new `middleware/pack_judge.py`**: `_PACK_CONTRACT`,
   `CURATOR_PACK_SPEC` (`prompt_id='curator_pack'`, `baseline_heuristics=_CURATOR_HEURISTICS`
   imported by reference from `task_curator.py`, harness version `curator-pack-v1` so a
   pack-contract edit never invalidates the single / batch artifacts), `CURATOR_PACK_OUTPUT_SCHEMA`,
   `PackVerdict`, `parse_pack_verdict`, the ONE `render_pack_member` (which imports
   `curator_types.clip_for_prompt`), and `judge_pack` as a function over a `TaskCurator` instance.
   `task_curator.py` does not grow and `followup_packing.py` imports nothing from the curator, so
   P0 does not collide with 6707. Verdict `pack | split | reject` with groups; the LLM writes prose
   only (title, description, details, justification per group). Items, files, survivor, priority
   and the declared-scope re-stamp are derived server-side. Rule 2 (narrow lock) gets its pack
   reading in `_PACK_CONTRACT`: for a `parent` pack the target lock set is the shared key set; for
   a `recipe` pack it is the union keys, already ≤ 15, and the question is whether every member
   applies the named recipe mechanically. The batch prompt is not reused: members are existing
   tasks, not candidates; the batch schema has no split; and a separate prompt_id keeps the
   artifact triple clean (PRD2-P §4). One pack per judge call in v1 (§9 item 3 for batching); the
   cost-ledger role is `curator_pack` (`_ledger_kwargs` takes any string; whether cost reports
   group a new role is P0's residual).
10. **Ledger = own sqlite `pack_sweep.db`** (`runs`, `proposals`, `outcomes`, `AUTOINCREMENT` ids,
    INV-13), the link-heal plan / apply / undo discipline, the single home of pack history.
    Observe mode writes ledger rows only: no store stamp (a stamp would put a non-decision in the
    store), no per-proposal event (the interceptor's `EventBuffer` feeds reconciliation), one log
    line per run. **Two keys, deliberately distinct.** The re-judge key `pack_key` is
    `sha256_16(sorted (task_id, text_fp))` — text only. The image a proposal is applied against is
    `(text_fp, dependencies, status, claimant stamp, holder edge present, item keys)` per member,
    NOT `updated_at`: every backend write path bumps `updated_at`, so a week of metadata-only
    churn (hold stamps, reconciliation stamps, `add_dependency`) would otherwise make every
    ledgered proposal `stale` under an unchanged `pack_key` and re-judge the whole ledger at the
    flip (PRD2-C b-2). A `pack_key` is never re-judged while unchanged and its last state is
    `proposed`, `rejected`, `split`, `applied`, `partial` or `undone`; a `stale` outcome whose
    members' `text_fp` and dependencies are unchanged is re-imaged on the next tick without a
    judge call. A second tick over an unchanged pool makes zero LLM calls.
11. **Loop host = the `TicketJanitor` shape** in `server/main.py`: in-process, so it reaches the
    interceptor's public methods and `_write_lock`, the curator's cap gate, breaker and cost ledger,
    and the known-project map. `CuratorConfig.pack_sweep: PackSweepConfig{mode: 'off' | 'observe' |
    'enforce' = 'off', interval_seconds = 86400, max_pack_items, max_union_keys,
    max_packs_per_run = 10, max_judge_calls_per_run = 25, write_failure_streak = 3}`; every leaf in
    `config/reload.py::RELOADABLE_FIELDS`, read per tick off the shared config object, never
    captured (the `write_triage.*` idiom) — including `interval_seconds`, which the pack loop
    re-reads before every sleep rather than taking it as a `run_loop` argument the way the janitor
    does (PRD2-C b-3). Deploy with `mode: observe` in `config.yaml`. `off` is the storm escape and
    needs no restart. Cadence from the re-run: at 25 judge calls a day the whole-pool stock (df
    118 packs, reify 134) is judged in 4.7 / 5.4 days, the reservation-eligible stock (64 / 112) in
    2.6 / 4.5 days; steady state is the inflow of new packs. First-pass cost upper bound at the
    measured $0.306 per call (a full-pool prompt; a 2–5-entry pack prompt is smaller): df ≈ $36,
    reify ≈ $41 over the whole-pool stock, ≤ $8 a day a project at the cap — the yardstick for Φ's
    cost series. `max_packs_per_run` is justified by reconciliation load: every member cancel is
    one targeted reconciliation run, so ≤ 10 packs × ≤ 4 cancels = ≤ 40 runs per project per tick
    (5197 restarted fused-memory at 164 transitions in one session).
12. **Executor order: pre-check live, re-read-and-rewrite the survivor under the lock, cancel
    members through the chokepoint's precondition, promote last.** The rewrite goes through one
    write half extracted from `_execute_combine` (`_write_combined_rewrite`, starting at the
    `details_with_files` build; the guard sequence stays in `_execute_combine`, which keeps its
    lock-free filing-time shape unchanged), so the `FILES_TO_MODIFY` block, the metadata merge, the
    item merge, the `declared_scope` re-stamp (basis `'pack'`, PRD 1 α's Literal gains the value;
    still one writer site), the audit row and the amendment advice have one copy. The pack path
    calls the half INSIDE `_write_lock(project_id)` after re-reading the survivor and comparing it
    to the proposal image (the discipline of the interceptor's public `update_task`), refusing
    `member_changed` / `member_claimed` / `member_not_held` on the survivor without writing. Members
    are cancelled through `_apply_status_transition`, which gains a `require_before` precondition
    evaluated against `before` inside its own critical section (never a wrapper that re-takes the
    non-reentrant lock); `cancel_superseded_if_pending` is a thin public wrapper that calls it
    directly, as the CSV branch does, and maps a typed refusal. That is the CAS against the
    scheduler's claim write: a claim lands first and the cancel refuses `member_claimed`, or it
    lands after and finds the task cancelled. 8143 edits the same function (gate + atomic
    `superseded_by`); P2 lands after it and adds the precondition on top of 8143's shape, adapting
    to whether 8143 threads `superseded_by` / `children` into `_apply_status_transition` or keeps
    them in `set_task_status`. Promotion runs last, so a half-applied pack is never ready. Partial
    outcomes are recorded with the executed steps; the next tick replays the remaining steps (each
    is idempotent) or, for a member claimed between rewrite and cancel, drops that member's item
    keys from the survivor in a second locked rewrite and records `partial(member_claimed)`.
13. **The rider reservation is owned by this PRD.** A pure predicate
    `followup_packing.rider_reserved(member, *, now) -> bool`: true iff the member has lock keys
    (a rider needs a file intersection) and `now − pending_since < RIDER_RESERVE_DAYS` (14) and
    `metadata.rider_ledger.declines` is below `RIDER_RESERVE_MAX_DECLINES`. The read contract this
    PRD fixes is minimal and the only thing it fixes: `rider_ledger` is a dict and `declines` an
    int; an absent key, or a present key without `declines`, reads as zero (so until the riders
    PRD lands the predicate is age-only). The riders PRD (PRD 3) owns the key — its blessing, its
    other fields, its writer — and inherits this read contract rather than guessing a name. The
    packer skips reserved tasks (`rider_reserved`) and carried tasks (`rider_of` present, `rider`).
    Fileless held tasks (reviewer suggestions) go straight to the packer; they can only ever reach
    c2. The riders PRD cites this constant and predicate and defines `rider_of`; it does not define
    a reservation of its own (§5). The reservation applies from P1 on, before the
    riders PRD lands, so the observe week measures the pool the enforce regime will see; the price
    is in decision 1 (about half of df's with-files members deferred in week one) and the report
    prints both coverage series.
14. **Undo is an interceptor method exposed as one admin MCP tool** (`undo_pack(proposal_id)`,
    the `replay_dead_letters` / `rebuild_candidate_key_index` precedent in `server/tools.py`),
    because reopening, re-holding and restoring a pre-image need the interceptor's lock and PRD 1's
    hold helper, which only the server process has. It reopens each member `cancelled → pending`
    with `reopen_reason='pack-undo:<proposal_id>'` — each comes back with a FRESH `pending_since`
    and a reset dispatch age, by `stamp_pending_since`'s D3 row; undo restores text, items and
    hold, not the clock — re-holds it through a new `followup_hold.rehold` (the inverse of
    `promote`: holder edge + `followup_hold` stamp + `followup_held` event), restores the survivor's
    title, description, details, priority, `followup_items` and `declared_scope` from the full
    pre-image stored in the ledger (the combine audit row truncates), re-holds the survivor, and
    writes `outcomes(state='undone')`. An undone `pack_key` is never re-proposed.
15. **The flip is a deterministic always-escalates gate task** (Φ) with a delayed milestone of 7
    days anchored on the observe leaf and the report leaf going done, whose L2 asks Leo to read
    the report and set `mode: enforce` (edit `config.yaml`, then `reload_config`), keep observing
    (re-arm), or set `off`. Because Φ depends on P3 → P2 → 8143, the flip cannot fire before the
    enforce code exists; the observe period is therefore ≥ 7 days and in practice until P2 lands.
    The first enforce tick applies ledgered `proposed` rows only after the executor's image check
    (decision 10), so a proposal whose members' text or dependencies changed is re-judged, never
    applied blind. Precedent: cancel-with-children β (observe) → ζ (flip) → η (retire). Here the
    flag has three states and stays: `off` is the kill switch.
16. **Reports render the no-producer state.** `pack_sweep_report.py` reads `sqlite_sequence` and
    prints `producer has not written` per table, distinct from zero (INV-13); each series the
    enforce regime adds (applied, partial, undone, survivor dispatch and landing within 30 d)
    prints the same marker until enforce has written. No starvation-watchdog series: a promoted
    survivor's watchdog clock starts at promotion (§2), so there is nothing to count.
17. **Nothing in 5197's residue is touched.** The 21 deferred members of finished carriers and
    the 91 behind pending carriers are legacy; their carriers are not held and do not pack (§8).
18. **A pack may promote held non-members.** PRD 1 decision 14's `promote` is a closure over the
    promoted task's held dependencies (`by='dependent'`). Under decision 8 every member shares
    those dependencies, so promoting the survivor promotes exactly the held tasks the pack already
    depended on, and row 14's `followup_promoted{by: 'pack'}` event may be accompanied by
    `by: 'dependent'` events for them (PRD2-C b-6).

## 4. Contract (B+H)

### 4.1 Candidate generation (`fused-memory/src/fused_memory/middleware/followup_packing.py`, pure; no I/O, no LLM, no curator import)

```python
PACK_MAX_ITEMS = MAX_FOLLOWUP_ITEMS          # from shared.followup_items (PRD 1 β); one home for the 5
PACK_MAX_UNION_KEYS = 15                     # the home of the curator's "≤ 15 files" budget (decision 6)
HUB_SHARE = 0.02                             # decision 5
RIDER_RESERVE_DAYS = 14                      # decision 13; cited by the riders PRD
RIDER_RESERVE_MAX_DECLINES = 3               # decision 13; §9 item 5

PackGenerator = Literal['parent', 'recipe']  # a2, c2; 'file_verb' is a v2 candidate (§8)
MemberKind = Literal['followup', 'review_suggestion']

@dataclass(frozen=True)
class PackMember:                            # built by the sweep from one task row; never from text parsing
    task_id: int
    title: str
    kind: MemberKind                         # source == 'agent-followup' → followup; spawn_context == 'review_suggestions' → review_suggestion
    parent: int | None                       # spawned_from
    lock_keys: tuple[str, ...]               # files_to_modules(strip_directory_locks(files), effective_lock_depth)
    item_count: int                          # len(followup_items) or 1 for a legacy single
    item_keys: tuple[str, ...]               # followup_items[].key, or (suggestion_hash,) for a legacy single
    band: str | None                         # declared_scope.band; None = not stamped (eligible)
    pending_since: datetime
    text_fp: str                             # sha256_16(title, description, details); the re-judge key
    status: str
    dependencies: frozenset[int]             # minus the holder edge
    held: bool                               # followup_hold present and promoted is None (the holder edge)
    is_holder: bool
    has_live_dependents: bool                # any live dependent or spawned_from child
    claimed: bool                            # claimant_run_id stamp present (the combine guard's reading)
    has_branch: bool                         # branch_base_sha
    has_plan: bool                           # plan_authored_at
    dedupe_skipped: bool                     # DEDUPE_SKIPPED_KEY present (6322 seam)
    rider_of: int | None                     # riders PRD; None on main today
    rider_declines: int                      # read contract: int(metadata.rider_ledger.declines); absent key or absent field reads 0;
                                             # the key, its blessing and its writer are PRD 3's (decision 13)
    rider_last_decline_reason: str | None    # metadata.rider_ledger.last_decline_reason when present; None = no hint (decision 7)

class PackExclusion(StrEnum):
    HOLDER = 'holder'; NOT_HELD = 'not_held'; BROAD = 'broad'; LIVE_DEPENDENTS = 'live_dependents'
    CLAIMED = 'claimed'; HAS_BRANCH = 'has_branch'; HAS_PLAN = 'has_plan'
    DEDUPE_SKIPPED = 'dedupe_skipped'; RIDER = 'rider'; RIDER_RESERVED = 'rider_reserved'

@dataclass(frozen=True)
class CandidatePack:
    generator: PackGenerator
    key: str                                 # parent id | recipe name
    members: tuple[int, ...]                 # 2..; sorted ascending
    union_keys: tuple[str, ...]
    survivor: int                            # survivor_of(members)
    pack_key: str                            # sha256_16(sorted (task_id, text_fp) pairs)

def rider_reserved(m: PackMember, *, now: datetime) -> bool                  # decision 13, pure
def exclusion(m: PackMember, *, now: datetime) -> PackExclusion | None      # decisions 7, 13; first match
def hub_keys(members: Sequence[PackMember], *, share: float = HUB_SHARE) -> frozenset[str]
def title_recipe(title: str) -> str | None   # ONE RECIPES table: h14_relocate, cite_doc_hygiene (v1), spot_dedupe, test_seam, diag_code
def survivor_of(members: Sequence[PackMember]) -> int                       # earliest pending_since, then lowest id
def candidate_packs(members: Sequence[PackMember], *, hubs: frozenset[str], now: datetime,
                    max_items: int = PACK_MAX_ITEMS, max_union_keys: int = PACK_MAX_UNION_KEYS) -> tuple[CandidatePack, ...]
    # eligible members only (exclusion is None); a2 then c2; disjoint. Within a group, partition by
    # `dependencies` (decision 8) first. a2: groups keyed (parent, kind), visited by smallest task_id;
    # union-find on shared non-hub keys; each component walked in ascending task_id. c2: one group per
    # recipe over the members a2 left; walked in the order of the tuple (directory_of_first_key, task_id),
    # directory_of_first_key = lock_keys[0].rsplit('/', 1)[0] over the member's lock_keys sorted
    # lexicographically (a root-level key with no '/' yields the key itself), '~' for a member with no
    # lock_keys so fileless members sort last. Chunking, both generators: before adding a member, close
    # the open chunk if it already holds max_items of item_count or the union of keys would exceed
    # max_union_keys; the member opens a new chunk; drop chunks of one. The pinned ordering is the one
    # the 2026-11-04 re-run used (pool dump header `c2_ordering`), so the fixture reproduces it.
    # Return order (the judge-call order, PRD2-P §6): (len(members) - 1) desc, oldest pending_since,
    # lowest id — not part of the fixture comparison, which is by set of (generator, member set).
```

Invariants: deterministic (same members, hubs and `now` → same packs, same order); every pack has
≥ 2 members, all eligible, all with the same non-holder dependency set and, for a2, the same
`parent` and `kind`; a member is in at most one pack; `survivor ∈ members`; no a2 edge runs
through a hub key. The module imports `shared.followup_items`, `shared.locking`,
`shared.declared_scope` (allowlisted reader, PRD 1 decision 8) and nothing from the interceptor or
curator (heuristic 13). Testable through the public signature alone.

### 4.2 Judge (`fused-memory/src/fused_memory/middleware/pack_judge.py`, new)

```python
_PACK_CONTRACT: str                          # inputs: one pack header + members; outputs: the schema below;
                                             # rule 2's pack reading (decision 9); "rewrite, not concatenate"
PACK_PROMPT_HARNESS_VERSION = 'curator-pack-v1'
CURATOR_PACK_SPEC = PromptSpec(prompt_id='curator_pack', contract=_PACK_CONTRACT,
                               baseline_heuristics=_CURATOR_HEURISTICS)   # imported from task_curator, by reference
CURATOR_PACK_OUTPUT_SCHEMA = {
  'verdict': 'pack' | 'split' | 'reject',
  'groups': [{'member_ids': [int, ...>=2], 'title': str, 'description': str, 'details': str, 'justification': str}],
  'justification': str }                     # pack-level; names the rule that fired on reject

class PackVerdict(BaseModel):                # frozen, extra='forbid'
    verdict: Literal['pack', 'split', 'reject']
    groups: tuple[PackGroup, ...]            # pack: exactly one group == all members; split: >= 1 proper
                                             # subsets, each >= 2, disjoint; reject: none
    justification: str                       # <= 500
    degraded: str | None                     # parser violation that forced reject, e.g. 'unknown-member', 'overlapping-groups'

def render_pack_member(m: PackMember, *, description: str, details: str, items: Sequence[FollowupItem]) -> str
    # the one home; clip_for_prompt, the same clipper PoolEntry.render uses; shows id, kind, title,
    # priority, spawned_from, band, lock keys, followup_items[].summary, clipped description and details
def parse_pack_verdict(raw: Mapping[str, Any], pack: CandidatePack) -> PackVerdict
    # validates member_ids against the pack (the _parse_decision_dict discipline); any violation
    # degrades to reject with `degraded` set, never raises
async def judge_pack(curator: TaskCurator, pack: CandidatePack, rendered_members: Sequence[str]) -> tuple[PackVerdict, JudgeCost]
    # one pack per call; invoke_with_cap_retry with curator._ledger_kwargs('curator_pack');
    # raises the DeferReason exceptions unchanged
```

The LLM never writes a survivor id, items, files or priority. The judge is reached only from the
sweep; the ticket worker never calls it. Members left out of a `split` stay held and unpacked. A
`degraded` reject is recorded in the ledger with the raw text, so a parser defect is visible in the
report (`no-silent-fail-soft`). `task_curator.py` is imported, never edited, by this module.

### 4.3 Ledger, loop and config (`middleware/pack_ledger.py`, `middleware/pack_sweep.py`, `config/schema.py`)

```python
class PackSweepConfig(BaseModel):            # CuratorConfig.pack_sweep; every leaf in RELOADABLE_FIELDS
    mode: Literal['off', 'observe', 'enforce'] = 'off'
    interval_seconds: float = 86400.0        # re-read before every sleep (decision 11)
    max_pack_items: int = PACK_MAX_ITEMS      # validated <= PACK_MAX_ITEMS
    max_union_keys: int = PACK_MAX_UNION_KEYS # validated <= PACK_MAX_UNION_KEYS
    max_packs_per_run: int = 10               # enforce: proposals applied per project per tick (reconciliation load)
    max_judge_calls_per_run: int = 25
    write_failure_streak: int = 3

class PackMemberImage(BaseModel):            # what the proposal was judged against; the apply precondition
    task_id: int; text_fp: str; dependencies: tuple[int, ...]; status: str
    claimed: bool; held: bool; item_keys: tuple[str, ...]

class PackProposal(BaseModel):               # one row per judged group; observe AND enforce
    proposal_id: int                         # AUTOINCREMENT
    run_id: int; project_id: str
    pack_key: str; generator: PackGenerator; key: str
    split_from: int | None                   # the parent proposal when this row is a split group
    survivor_id: int
    members: tuple[PackMemberImage, ...]
    union_keys: tuple[str, ...]
    rewrite: RewrittenTask | None            # curator_types shape; None on reject
    items: tuple[FollowupItem, ...]          # merge_items over members (legacy_item for singles); derived
    verdict: Literal['pack', 'split', 'reject']
    justification: str; degraded: str | None
    judge_model: str; harness_version: str; cost_usd: float
    survivor_preimage: dict | None           # full title/description/details/priority/followup_items/declared_scope; written at apply
    mode_at_proposal: Literal['observe', 'enforce']

class PackOutcome(BaseModel):                # append-only; latest row per proposal is its state
    proposal_id: int
    state: Literal['proposed', 'rejected', 'split', 'stale', 'refused', 'applied', 'partial', 'undone']
    refused_reason: PackRefusal | None
    executed: tuple[PackStep, ...]           # {step: 'rewrite' | 'cancel' | 'promote', task_id, ok, error}
    at: str

class PackRefusal(StrEnum):
    MEMBER_CHANGED = 'member_changed'; MEMBER_NOT_HELD = 'member_not_held'; MEMBER_CLAIMED = 'member_claimed'
    MEMBER_HAS_BRANCH = 'member_has_branch'; MEMBER_HAS_PLAN = 'member_has_plan'; MEMBER_NOT_PENDING = 'member_not_pending'
    MEMBER_HAS_LIVE_CHILDREN = 'member_has_live_children'; DEPENDENCY_MISMATCH = 'dependency_mismatch'
    HOLDER = 'holder'; DEDUPE_SKIPPED = 'dedupe_skipped'; RIDER = 'rider'

class PackLedger:                            # $DARK_FACTORY_DATA_DIR/pack_sweep.db; CREATE TABLE IF NOT EXISTS; AUTOINCREMENT
    def start_run(project_id, mode) -> int
    def finish_run(run_id, *, judged, proposed, rejected, applied, refused, stopped_reason: str | None)
    def known_pack_keys(project_id) -> Mapping[str, str]          # pack_key -> latest state (the no-re-judge rule)
    def record_proposal(p: PackProposal) -> int
    def reimage(proposal_id, members: Sequence[PackMemberImage]) -> None   # stale → proposed without a judge call (decision 10)
    def record_outcome(o: PackOutcome) -> None
    def applicable(project_id, limit) -> list[PackProposal]       # state 'proposed' and verdict 'pack' (incl. split groups), oldest first
    def undo_target(proposal_id) -> PackProposal                  # state 'applied' or 'partial'
    def producer_has_written() -> Mapping[str, bool]              # sqlite_sequence per table (INV-13)

class PackSweep:                             # TicketJanitor shape, except the interval is read per sleep
    async def run_loop(self)                                      # sleep(config.pack_sweep.interval_seconds), tick, except Exception: logger.exception
    async def tick(self)                                          # every project in the known-project map, each step wrapped
```

Tick, per project: (1) read `mode` off the shared config; return on `off`; skip the project if it
has pending curator tickets or the ZOT breaker is open (filing-time curation keeps the account pool
first). (2) Build `PackMember`s from the held pool (`followup_hold` present, `promoted is None`),
compute `hubs`, `candidate_packs(now=...)`. Then, in this order: first `reimage` every `stale`
proposal whose members' `text_fp` and dependencies are unchanged (its `pack_key` is unchanged by
construction; the row goes `stale → proposed` with no judge call); then drop any `pack_key` whose
latest state is `proposed`, `rejected`, `split`, `applied`, `partial` or `undone` — a `stale` row
that `reimage` could not restore (text or dependencies changed) has a different `pack_key` now and
is judged as a new candidate. (3) Judge in the
generator's order up to `max_judge_calls_per_run`; on a `DeferReason` exception end the tick and
record `stopped_reason`; the safe degrade is "no proposal", never "pack unjudged". (4) Record
proposals and outcomes (`proposed` / `rejected` / `split` + one `proposed` row per split group). In
observe mode, stop. (5) Enforce: apply at most `max_packs_per_run` applicable proposals, oldest
first, through `apply_pack`; after `write_failure_streak` consecutive `partial` outcomes stop and
file one folded escalation (`file_folded_escalation`, anchor `pack-sweep-failure:<project_id>`).
Ledger I/O runs through `asyncio.to_thread`; the judge call is already awaited; nothing blocks the
loop thread per member (`loop-thread-occupancy-bounded`).

Invariants: a tick in observe mode performs no store write (test: row hash of `tasks.db` before
and after); a second tick over an unchanged pool records a run with `judged = 0`; `runs` gets a row
even when the tick stops early (the `stopped_reason` is the fact, not a log line); every proposal
row has an outcome row; every config leaf including the interval takes effect at the next tick or
sleep with no restart.

### 4.4 Executor and undo (`middleware/pack_executor.py`, `task_interceptor.py`, `followup_hold.py`)

```python
# task_interceptor.py
async def TaskInterceptor._apply_status_transition(self, *, ..., deferral_call,
                                                   require_before: Callable[[dict], PackRefusal | None] | None = None) -> dict
    # NEW kwarg: evaluated against `before` inside the existing `async with self._write_lock(project_id)` block,
    # right after the before-read; a non-None refusal returns without writing (typed, journaled)
async def TaskInterceptor.cancel_superseded_if_pending(self, task_id: int, *, superseded_by: Sequence[int],
                                                       cause: str, project_root: str) -> PackRefusal | None
    # thin public wrapper: builds the DeferralCall and calls _apply_status_transition('cancelled', require_before=
    # lambda before: MEMBER_NOT_PENDING if before.status != 'pending' else MEMBER_CLAIMED if claimant stamp else None,
    # superseded_by=..., children={} as 8143 threads them); never takes _write_lock itself
async def TaskInterceptor._write_combined_rewrite(self, *, target: dict, rewrite: RewrittenTask, items: Sequence[FollowupItem],
                                                  marker: Mapping[str, Any], basis: Literal['combine', 'pack'],
                                                  audit: Mapping[str, Any], project_root: str) -> dict | None
    # the write half of _execute_combine from the details_with_files build on: FILES_TO_MODIFY block,
    # metadata_mode='merge' of {curator_action, curator_justification[:500], combined_at} + marker + followup_items,
    # the declared_scope re-stamp with `basis`, the audit row (gains pack_proposal_id, members), the amendment advice.
    # Takes no lock: _execute_combine calls it lock-free as today; the pack path calls it inside _write_lock.
async def TaskInterceptor.rewrite_pack_survivor(self, proposal: PackProposal, *, project_root: str) -> PackRefusal | None
    # async with self._write_lock(project_id): re-read the survivor; refuse MEMBER_CHANGED / MEMBER_CLAIMED /
    # MEMBER_NOT_HELD / MEMBER_NOT_PENDING against proposal.members[survivor]; store the pre-image on the ledger
    # row; _write_combined_rewrite(basis='pack', marker={'pack_record': {...}}, items=proposal.items)
async def TaskInterceptor.undo_pack(self, proposal_id: int, *, project_root: str) -> PackOutcome

# pack_executor.py
async def apply_pack(interceptor, ledger, proposal: PackProposal, *, project_root) -> PackOutcome
    # 1. pre-check every member live, no lock: image (text_fp, dependencies, status, claimed, held, item_keys) ==
    #    proposal; eligibility (no branch, no plan, not holder, not dedupe_skipped, not a rider); equal non-holder
    #    dependency sets; no live dependent or spawned_from child. Any miss → outcome 'stale' (image) or 'refused'
    #    (eligibility), nothing written; a stale pack is re-imaged or re-judged later (decision 10).
    # 2. interceptor.rewrite_pack_survivor(proposal)        # re-read + write under the lock; refusal → 'refused', nothing written
    # 3. for each non-survivor member in ascending id: cancel_superseded_if_pending(member, superseded_by=[survivor])
    # 4. promote(survivor, by='pack', ref=f'pack:{proposal_id}')   # PRD 1 ζ; last, so a half-applied pack is never ready;
    #    may cascade by='dependent' onto the pack's shared held dependencies (decision 18)
    # 5. outcome 'applied' with the steps; on any failed step stop and record 'partial'

# followup_hold.py (PRD 1 module; P2 adds the inverse)
def rehold(tm, project_root, task_id: int, *, ref: str) -> None
    # holder edge + followup_hold stamp (promoted=None) + followup_held event; refuses a non-pending task
```

`pack_record{proposal_id: int, generator, members: list[int], packed_at}` is a registered
submodel (`register_metadata_submodel`, cardinality `dict`), written only by
`_write_combined_rewrite` under `basis='pack'`. Replay of a `partial`: the next tick re-runs
`apply_pack` on the same proposal; the rewrite is replace-by-content behind the same image check,
a cancel of a cancelled member is a same-status no-op, promote is idempotent on a non-held task.
The one non-convergent case, a member claimed between steps 2 and 3, is closed by dropping that
member's item keys from the survivor in a second locked rewrite (the keys are known from `items`)
and recording `partial(member_claimed)`; that member stays held and unpacked. `undo_pack`: for
each cancelled member `set_task_status(member, 'pending', reopen_reason='pack-undo:<id>')` (fresh
`pending_since` by D3) then `rehold`; restore the survivor from `survivor_preimage` through the
backend `update_task` (replace mode, under the lock) and `rehold` it; record `undone`. The pack
audit row in `combine_audit.jsonl` carries `pack_proposal_id` and `members` and keeps its truncated
fields; the ledger row is the home of the pre-image (`one-fact-one-home`).

### 4.5 Boundary-test sketch (the integration gate's signal, P3; every scenario drives `PackSweep.tick`, the public interceptor methods, `get_task`, `set_task_status`, `undo_pack`, `reload_config` and the report script, never internals)

| # | scenario | preconditions | postconditions |
|---|---|---|---|
| 1 | observe tick over a pool with two coherent a2 packs and one c2 pack | `mode: observe`; judge stubbed at the LLM boundary to `pack` | `runs` 1 row, `proposals` 3 rows with `generator` and `survivor_id` = earliest `pending_since`; `tasks.db` row hash unchanged; no `task_modified` event |
| 2 | second tick, pool unchanged | scenario 1 | `judged == 0`, zero LLM calls, one more `runs` row |
| 3 | a member's description edited between ticks; another member gets a metadata-only stamp | scenario 1 | the first pack's `pack_key` changes and is judged again (old proposal stays `proposed`); the second pack is NOT re-judged (same `pack_key`) and its image is refreshed at apply time, not before |
| 4 | judge raises `AllAccountsCappedException` on call 2 of 3 | scenario 1 shape | tick ends; `runs.stopped_reason == 'all_accounts_capped'`; call 1's proposal recorded; call 3 not attempted |
| 5 | `max_judge_calls_per_run = 1` | 3 candidate packs | one proposal; the next tick judges the next pack in generator order (largest first, then oldest) |
| 6 | pool contains the holder, a `broad` member, a member with a live dependent, a claimed member, a member with `branch_base_sha`, one with `plan_authored_at`, one with `dedupe_skipped`, one with `rider_of` | all share a parent and a non-hub key | none is a pack member; the report's exclusion counts name each `PackExclusion` |
| 7 | five same-parent follow-ups, three behind Ω-style extra dependency `[D]`, two with none | same non-hub key | two packs (3 + 2), never one of 5; `dependency_mismatch` never arises at apply |
| 8 | two 3-item bundles with the same parent and key | PRD 1 bundles | no pack (Σ items 6 > 5); a 3-item bundle and a 2-item bundle do pack |
| 9 | a key declared by 3% of the pool | a2 candidates sharing only that key | no a2 edge through it; a pair that also shares a non-hub key still packs |
| 10 | a held task with files, `pending_since` 10 days ago; another 15 days ago; a fileless reviewer suggestion 1 day ago; a 10-day task whose `rider_ledger` shows 3 declines | riders reservation | first excluded `rider_reserved`; the other three eligible (the suggestion only through c2) |
| 11 | judge returns `split` into {A,B} and {C,D}, leaving E | 5-member a2 pack | parent row `split`; two `proposed` rows with `split_from`; E stays held and unpacked |
| 12 | judge returns a group naming a non-member id | any | verdict degrades to `reject` with `degraded == 'unknown-member'`; raw text in the ledger |
| 13 | `reload_config` after editing `mode: observe → enforce` and `interval_seconds` | server running | `applied` lists both `curator.pack_sweep.*` leaves; next tick enforces and the next sleep uses the new interval; no restart |
| 14 | enforce tick applies a `proposed` pack of three (PRD 1 §4.4 row 19) | 8142, 8143 landed; members held | survivor: items = union, `pack_record`, `declared_scope.basis == 'pack'`, no holder edge, `followup_promoted{by: 'pack'}` (plus `by: 'dependent'` for any shared held dependency); two members `cancelled` with `superseded_by == [survivor]`; zero L1s and zero residue records; outcome `applied` |
| 15 | a member promoted by citation and claimed by the scheduler (`set_task_status(in-progress, claimant_run_id)`) between judgement and apply, including a claim that lands while `apply_pack` is between steps 2 and 3 | enforce | pre-check refuses `member_claimed`; or `_apply_status_transition`'s precondition refuses inside the lock, the survivor's second locked rewrite drops that member's keys, outcome `partial(member_claimed)`; the claimed task is never cancelled |
| 15b | the SURVIVOR is promoted by citation and claimed between judgement and apply | enforce | `rewrite_pack_survivor` refuses `member_claimed` under the lock; nothing written; outcome `refused` |
| 16 | a member's text edited after judgement | enforce | outcome `stale`; nothing written; the pack is re-judged next tick under its new `pack_key` |
| 16b | a member received only a metadata stamp after judgement | enforce | image still matches (text, dependencies, status, claimant, hold, item keys unchanged); the pack applies |
| 17 | the promote step fails (injected) after rewrite and cancels | enforce | outcome `partial` with three ok steps and one failed; the survivor is still held (not ready); the next tick completes it and records `applied` |
| 18 | three consecutive `partial` outcomes | enforce | tick stops; one folded escalation at anchor `pack-sweep-failure:<project>`; `runs.stopped_reason == 'failure_streak'` |
| 19 | `undo_pack(proposal_id)` on scenario 14 | applied | members `pending` with `reopen_reason` set, a fresh `pending_since`, and the holder edge back; survivor restored to its pre-image text and items and held; outcome `undone`; the `pack_key` is never re-proposed |
| 20 | report over an empty ledger; over scenario 1; over scenario 14 | — | prints `producer has not written` per table; then observe series (both coverage series, exclusions, recipes) with enforce series marked unwritten; then all series populated, survivor dispatch within 30 d counted |
| 21 | `mode: off` | any pool | tick returns at once; no `runs` row; no LLM call |
| 22 | a filing-time combine after P2 | ticket candidate, pending target | `_execute_combine` behaves as before P2 (lock-free guards, same write, items merged, `basis == 'combine'`); `test_task_interceptor_curator.py` green |

## 5. Cross-PRD relationship and seams (G4)

| other PRD / document | direction | seam mechanism | owner | status |
|---|---|---|---|---|
| `plans/follow-up-filing-shape-prd.md` (PRD 1; 8413–8421 pending) | consumes | `followup_hold` record and holder (ε 8417) as the pool predicate; `followup_items`, `merge_items`, `legacy_item`, `MAX_FOLLOWUP_ITEMS` (β 8414); the item-preserving `_execute_combine` and audit fields (γ 8415) that P2 extracts the write half from (P2 moves γ's item merge and re-stamp, it does not re-author them); `promote(by='pack')` and its `by='dependent'` cascade (ζ 8418); `declared_scope.band` as a reader (α 8413, allowlist gains `followup_packing.py`, `pack_sweep.py`, citing this PRD) and `DeclaredScope.basis` gaining `'pack'` (P2 edits α's Literal; one writer site kept) | PRD 1 owns the records; this PRD owns the packer, the extraction, the `'pack'` basis value, `rehold`, and boundary row 19 (PRD 1 §4.4 defers it here). Companion correction at decompose: PRD 1 decision 15 and row ε's waiver text (task 8417's `g7_waivers` rationale) change from "the drain is the packing sweep PRD" to "the packing sweep and riders are the drains, bounded by θ's report" | queued |
| `plans/cancel-with-children-prd.md` (8142–8148 pending) | consumes | `superseded_by` Tier-A (α 8142) written atomically with the cancel (β 8143); the gate runs on every member cancel with `children={}`; enforce before ζ 8147 is acceptable because packs carry no children by construction (pre-check); P2 adds `require_before` on top of 8143's edit to `_apply_status_transition` and adapts to 8143's threading of `superseded_by` / `children` | that PRD owns the vocabulary and gate; this PRD owns the precondition and `cancel_superseded_if_pending` and adds a "pack executor" row to its §4.14 table as a companion correction | queued |
| `plans/followup-riders-prd.md` (PRD 3, option C; committed `c393a67c54`, filed as 8438 α, 8439 β, 8440 γ, 8441 δ) | produces for / consumes | `RIDER_RESERVE_DAYS`, `RIDER_RESERVE_MAX_DECLINES`, `rider_reserved` and the `rider_reserved` / `rider` exclusions live here; PRD 3 defines `rider_of` (set only on a carried rider, cleared inside the host's done / cancel transaction and by its 14-day carry sweep on a stalled host — the definition this PRD's `RIDER` exclusion names, decision 7) and owns `metadata.rider_ledger` (blessing, fields, writer: `offers`, `declines`, `last_host`, `last_decline_reason`, `carried_at`, …) under this PRD's read contract (`declines: int`, absent reads 0; `last_decline_reason` as a same-parent pack hint, absent reads no hint; decisions 7, 13); it cites this PRD's constant and predicate rather than defining its own and never carries a task this PRD has packed (a cancelled member is terminal; a survivor is promoted and no longer held). PRD 3 γ (8440) will depend on this PRD's P1 once P1 is filed (the edge is added at this PRD's decompose) | this PRD owns the reservation and the `exclusion()` reads; PRD 3 owns `rider_of` and `rider_ledger` | wired (PRD 3 committed); edge 8440 → P1 queued for decompose |
| task 6322 (re-dedupe sweep over `dedupe_skipped`; pending low) | coexists | the packer excludes `DEDUPE_SKIPPED_KEY` tasks until 6322 clears the marker; 6322 may run as a second arm of `PackSweep`'s loop host (one curator-sweep loop per server) | 6322 | pending; no edge |
| task 5491 (`sweep_zot_duplicate`) | precedent | fail-safe rule reused: an errored read is never evidence; the ZOT breaker is shared and the tick skips while it is open | — | wired |
| task 4503 (cap deferral, `DeferReason`) | consumes | `runs.stopped_reason` vocabulary; a capped tick stops, nothing parks | — | wired |
| task 5197 (backlog coalesce sweep, done) | precedent | files-gated membership, gate/branch exclusions carried forward; cancel-with-pointer instead of defer, a durable ledger instead of `/tmp`, in-place survivor instead of a carrier; its residue untouched (decision 17) | — | n/a |
| tasks 6707 / 6708 (curator restructure; 6707 pending high) | coexists | P0 imports `_CURATOR_HEURISTICS` and `TaskCurator` from `task_curator.py` and edits nothing there; if 6707 renames either, P0 rebases its import | no edge | none needed |
| reify alpha-spine Ω 9739 | coexists | the Ω edge is an ordinary non-holder dependency: members behind Ω pack only with members behind Ω (decision 8); promotion removes the holder edge only, so a survivor stays behind Ω | reify spine PRD | wired |
| task 4827 (task sizing recalibration, deferred) | produces for | packed survivors' `declared_scope_outcome` events (PRD 1 η) carry `basis='pack'` | 4827 | n/a |

No reciprocal-ownership statements found.

## 6. Decomposition plan

Overlay bands: 300–1,500 LOC and ≤ 10–12 files per leaf. The brief's P1 (ledger + loop +
generators + judge + observe) would be about 14 files and 1,800–2,200 LOC, over both bounds, so the
generators and the judge are split out as P0 (pure, LLM-boundary-only), as the brief allows.
Labels are P0–P3 and Φ; ids are assigned at decompose. Same-file leaves are serialized by edges
(`task_interceptor.py` in P2 only; `config/schema.py`, `reload.py`, `main.py`, `config.yaml` in P1
only; `followup_hold.py` and `declared_scope.py` in P2 only, after PRD 1 ζ and γ; `task_curator.py`
in no leaf).

| label | title | modules | observable signal | prereqs |
|---|---|---|---|---|
| P0 | Pack candidate generators, hub rule, exclusions, rider reservation, survivor rule, and the pack judge | `fused-memory/src/fused_memory/middleware/followup_packing.py` (new: §4.1), `fused-memory/tests/test_followup_packing.py` (new: determinism, a2/c2, hubs, exclusions, `rider_reserved`, dependency partition, item and key bounds, survivor rule, recipe table), `fused-memory/src/fused_memory/middleware/pack_judge.py` (new: §4.2), `fused-memory/tests/test_pack_judge.py` (pack spec resolves under `curator-pack-v1`; parser degrade cases; heuristics shared by identity; `judge_pack` through a stubbed `invoke_with_cap_retry`), `fused-memory/tests/fixtures/pack_pool_2026-11-04.json` (a verbatim copy of the committed `plans/followup-packing-sweep-prd.pack-pool-2026-11-04.json`, made by this leaf — the plans/ file is committed with the PRD, the tests tree is touched only by the leaf) and `fused-memory/tests/fixtures/packs-a2c2-2026-11-04.csv` (copy of `plans/followup-packing-sweep-prd.packs-a2c2.csv`, the expected output), `shared/tests/test_declared_scope_readers.py` (allowlist + `followup_packing`, citing this PRD) | intermediate: unlocks P1. A fixture test builds `PackMember`s from every record in the pool dump with every field the dump lacks forced to the no-exclusion value (`dependencies` EMPTY for every member, so decision 8's partition is exercised only by the synthetic row 7; `held` true, `is_holder`, `claimed`, `has_branch`, `has_plan`, `dedupe_skipped` false, `rider_of` None, `rider_declines` 0, `has_live_dependents` false, `band` None, `text_fp` from the title, because PRD2-P §0 applied no exclusions), computes hubs at share > 2% (expect the header's 8 / 16 keys) and runs `candidate_packs(now = 2026-11-04T00:00Z)` twice — once with `rider_reserved` disabled, once enabled — and reproduces the CSV as a set of (generator, member set) pairs per run, under the c2 ordering pinned in §4.1 and the dump header: whole pool df 118 packs (a2 77, c2 41), reify 134 (a2 89, c2 45); minus reservation df 64 (34, 30), reify 112 (76, 36); `PromptArtifactStore.resolve(CURATOR_PACK_SPEC, 'sonnet', 'curator-pack-v1')` returns the in-code baseline | PRD 1 β 8414 (`MAX_FOLLOWUP_ITEMS`, `FollowupItem`), α 8413 (`DeclaredScope`) |
| P1 | Pack ledger, sweep loop and config; observe mode deployed | `middleware/pack_ledger.py` (new: §4.3 tables, `reimage`, `producer_has_written`), `middleware/pack_sweep.py` (new: member assembly from the store incl. `rider_ledger` and item keys, hubs, tick, re-image, `DeferReason` stop, folded escalation, per-sleep interval read, start-log line), `config/schema.py` (`PackSweepConfig` on `CuratorConfig`, bound validators), `config/reload.py` (every `curator.pack_sweep.*` leaf with the reload-safety note), `server/main.py` (build and start beside the janitor; cancel in shutdown), `fused-memory/config/config.yaml` (`curator.pack_sweep: {mode: observe, ...}`), `shared/tests/test_declared_scope_readers.py` (+ `pack_sweep`), tests: `test_pack_ledger.py`, `test_pack_sweep.py` (rows 1–5, 11–13, 21 with the judge stubbed at the LLM boundary; `tasks.db` row-hash invariant) | leaf (observe): after deploy the start log shows `Pack sweep: mode=observe`; `pack_sweep.db` gains a `runs` row per project per day and `proposals` rows for the real held pool; a second tick makes zero judge calls; `tasks.db` is unchanged across a tick | P0; PRD 1 ε 8417 (held pool, holder) |
| P2 | Pack executor, status-chokepoint precondition, locked survivor rewrite, combine write-half extraction, undo | `middleware/pack_executor.py` (new: `apply_pack`, the partial replay, the second-rewrite drop), `task_interceptor.py` (`require_before` on `_apply_status_transition`, added on top of 8143's edit and adapted to its threading; `cancel_superseded_if_pending`; extract `_write_combined_rewrite` from the `details_with_files` build on and call it from `_execute_combine` unchanged in behaviour; `rewrite_pack_survivor` under the lock; `undo_pack`; `_append_combine_audit` gains `pack_proposal_id`, `members`), `middleware/pack_sweep.py` (enforce arm of the tick: `applicable`, `max_packs_per_run`, failure streak), `followup_hold.py` (`rehold`), `shared/src/shared/declared_scope.py` (`basis` Literal + `'pack'`), `shared/src/shared/task_metadata.py` (register `pack_record`), `server/tools.py` (`undo_pack` tool), `docs/task-authoring.md` §8 (`pack_record`), `plans/cancel-with-children-prd.md` §4.14 (one "pack executor" row), tests: `test_pack_executor.py` (rows 14–19, 15b, 16b, 22 against a real sqlite stack, incl. the claim-race tests that drive `set_task_status(in-progress, claimant_run_id)` through the interceptor concurrently with `apply_pack`, for a member and for the survivor), `test_task_interceptor_curator.py` (combine still green through the extracted half) | intermediate: unlocks P3 and Φ. With `mode: enforce` in a test stack, an applied proposal leaves one survivor (items = union, `pack_record`, `basis == 'pack'`, promoted `by == 'pack'`) and N−1 members `cancelled` with `superseded_by == [survivor]`, zero L1s; a claim racing either step is refused inside the lock; `undo_pack` restores the pre-image | P1; PRD 1 γ 8415, ζ 8418; cancel-with-children α 8142, β 8143 |
| P3 | Observe report and the integration gate | `scripts/pack_sweep_report.py` (new, read-only over `pack_sweep.db`, `tasks.db` and `runs.db`: the §1 series, coverage over the reservation-eligible pool AND the whole held pool, exclusion counts by `PackExclusion`, recipe hit counts, seeded proposal sample, survivor dispatch and landing within 30 d, every series marked when its producer has not written), `tests/scripts/test_pack_sweep_report.py`, `fused-memory/tests/integration/test_pack_sweep.py` (new; every §4.5 row through public interfaces; the directory is PRD 1 ι 8420's home — create it here if P3 lands first) | leaf (signal): the report printed over the live ledger after ≥ 1 observe day shows real proposals, both coverage series, and `producer has not written` only on the enforce series; §4.5 rows 1–22 green against the integrated tree | P2 |
| Φ | Flip gate: read the observe report, set `mode: enforce` (deterministic, always-escalates, delayed milestone 7 d) | no code: `task_kind='deterministic'`, `always_escalates=true`, `execution_class='decision'`, `metadata.milestone: {mode: 'delayed', after_secs: 604800}` anchored on P1 and P3 going done (`milestone_deps_satisfied_at`, frozen once; ≥ 7 observe days of real rows exist when it fires, and P3 → P2 means the enforce code exists); its description names `scripts/pack_sweep_report.py`, the three rulings (enforce / keep observing and re-arm / off), the reload step (`fused-memory/config/config.yaml` then `reload_config`), the a2 + c2 coverage figures from decision 1 (both series) and the reject-rate yardstick (PRD2-P §2: expect ~40% reject + split on a2, ~20% on c2; a2 above 60% means tighten before enforcing), and the cost yardstick (decision 11) | leaf: one L2 on the dark-factory queue carrying the report path; after Leo's ruling, `runs.mode == 'enforce'` rows appear (or the gate is re-armed) | P1, P3 |

Companion corrections at decompose: the cancel-with-children PRD §4.14 row is P2's edit (no
separate task); PRD 1 §4.4 row 19's "future" status needs no edit (it already defers to this gate);
PRD 1 decision 15 and row ε, and task 8417's stamped `g7_waivers` rationale, are amended to "the
packing sweep and riders are the drains, bounded by θ's report" (a note on 8417 plus the PRD 1
text edit, committed with the decompose record). P0 and P2 carry `metadata.g7_waivers: []` and no
`x_finding_key` (this PRD remediates no instrument finding; the overlay's trigger-chain gate does
not apply).

G7 walk (walked in author mode; blocking at decompose, re-walked there). `storm-escape-required`:
`mode: off` (hot), `max_judge_calls_per_run`, `max_packs_per_run`, the `write_failure_streak` stop
with one folded escalation, the `DeferReason` stop. `holds-owned-and-bounded`: this PRD introduces
no new hold; members stay under PRD 1 ε's waived hold (whose rationale the companion correction
narrows to a partial drain), a `partial` pack's members are still held and owned by the next tick's
replay bounded by the streak, and a `proposed` row is a ledger fact, not a hold on a task. No
waiver. `loop-thread-occupancy-bounded`: ledger I/O through `asyncio.to_thread`, per-run caps, one
project at a time. `corroborate-before-acting`: the live pre-check, then the survivor re-read under
the write lock immediately before its write, then the member precondition inside the status
chokepoint's critical section; a stale image is never applied, for members and for the survivor
alike. `status-matches-liveness`: every cancel and the survivor rewrite happen under the write lock;
promote last. `no-silent-fail-soft`: `stale` / `refused` / `partial` / `degraded` are typed and
ledgered; `stopped_reason` is a row, not a log line. `structured-facts-at-failure`: `PackRefusal`,
`PackExclusion`, `executed` steps. `one-fact-one-home`: pack history in the ledger only; the 5 in
`followup_items`, the 15 in `followup_packing`, hubs computed not configured, the reservation
constant and predicate in one module cited by the riders PRD; the pre-image in the ledger with a
pointer in the audit row; `render_pack_member` in `pack_judge.py` only. `no-lockstep-duplication`:
one write half for combine and pack; heuristics shared by reference; the clipper shared.
`contracts-machine-checked`: pydantic models, the output schema, StrEnums, bound validators.
`guards-exercise-behaviour`: the race tests drive the real claim path against both the member
cancel and the survivor rewrite; the observe test hashes `tasks.db`. `exceptions-owned-or-ratified`:
the two declared-scope allowlist entries cite this PRD and their task ids; no suppression added.
`readers-prove-their-producer`: P3 and Φ (decision 16). `sampler-vocabulary-one-table`: not touched.

## 7. Pre-conditions

- No novel substrate on main: every mechanism extends a site named in §2 (G3 verified against
  `8d45e1edca`, symbol by symbol, 2026-11-04). The substrate this PRD assumes from PRD 1
  (`followup_hold.py`, `followup_items.py`, `declared_scope.py`, `promote`, the item-preserving
  combine) is filed and pending as 8413–8418 and wired as hard prerequisites above; nothing here
  dispatches before the leaf it needs.
- P2 needs cancel-with-children 8142 and 8143 (pending, chained 8142 → 8143, no other edges).
  Observe mode (P0, P1) does not. The observe period is ≥ 7 days and in practice until P2 lands.
- `mode: observe` ships in `config.yaml` with P1 and takes effect at the next fused-memory
  redeploy (the fleet sweep) or an operator `reload_config`. No operator action is required for
  observe; enforce is Φ's human act.
- `DARK_FACTORY_DATA_DIR` resolves as it does for `combine_audit.jsonl`; the ledger file is created
  on first start.

## 8. Out of scope

- An age-out for held tasks (Leo: no; decision 1). The question reopens at PRD 1 θ 8421 with numbers.
- The title-verb generator b (`'file_verb'`), a v2 candidate: PRD2-P measured it at df 47 / reify
  53 avoided landings on the 11-04 stock and wrong in 2 of 5 readings; revisit once the observe
  report shows a2 and c2 reject rates and whether a closed verb lexicon is worth maintaining.
- Dispatch-time riders (option C): a separate PRD; the seam and the reservation's ownership are §5.
- Raising survivor priority on pack (decision 3): reopened only by P3's 30-day dispatch and landing
  series.
- A starvation-watchdog series or mitigation: no premise (§2, decision 16).
- Deriving `files` for reviewer suggestions from their `[category] path:line` titles (an ad-hoc
  parser, heuristic 12); the fix belongs at `review_suggestions/curator_intake.py`, not here.
- 5197's residue (21 + 91 deferred members) and the legacy `x_coalesced_into` readers.
- Batching several packs into one judge call (§9 item 3).
- Packing anything outside the held pool: medium-or-higher tasks, PRD leaves, cited follow-ups.
- Per-project `mode` overrides; the server-wide flag is enough for one observe period (§9 item 4).

## 9. Open questions (tactical)

1. **Boundedness and age-out.** Stated in decision 1 from the re-run (whole pool df 24% / reify 35%
   before the judge, ≈ 11% / 16% after; reservation-eligible pool df 14% / ≈ 7%, reify 29% /
   ≈ 13%); the pool is not bounded by packing alone. Not reopened here; PRD 1 θ (8421) reopens it
   with the 14-day numbers. **Suggested resolution:** none in this PRD; Φ's L2 quotes both series.
2. **Recipe table v1 contents.** P0 ships `h14_relocate`, `cite_doc_hygiene` (tightened),
   `spot_dedupe`, `test_seam`, `diag_code` from PRD2-P. `diag_code` is reify-only (2 packs / 8
   avoided in the re-run) and `test_seam` is small (df 6 / 16, reify 2 / 4); whether they stay is
   decided from the observe report's recipe hit counts. **Suggested resolution:** keep; drop any
   recipe with zero accepted packs in the observe period.
3. **Judge batching.** One pack per call in v1 (decision 11's cost yardstick). If the observe
   period shows cost matters, batch `batch_max` packs per call behind the same parser. Decide at
   P3 from the `cost_usd` series.
4. **Per-project mode.** A server-wide `mode` means df and reify flip together. **Suggested
   resolution:** keep server-wide for v1; add `PackSweepConfig.project_overrides` only if Φ's
   ruling differs by project.
5. **`RIDER_RESERVE_MAX_DECLINES = 3`.** The value at which a declined rider leaves the
   reservation early is set here because the predicate's home is here, but no `rider_ledger`
   writer exists until the riders PRD lands, so until then the predicate is age-only. The riders
   PRD's author may propose a different value with evidence; the constant's home does not move.
6. **Integration test directory.** `fused-memory/tests/integration/` is created by PRD 1 ι 8420 or
   by P3, whichever lands first; the second to land rebases (no edge, same treatment as 6707).
7. **Cost-report grouping of the `curator_pack` role.** `_ledger_kwargs` accepts the string; whether
   the invocations ledger's reports group an unknown role is checked in P0 and, if not, the role is
   added where the report enumerates roles.

## Appendix A. G1 / G3 / G6 self-check

- G1: every mechanism in §1's table names a consumer that is either wired on main (`server/main.py`
  startup, `reload_config`, `PromptArtifactStore.resolve`, the steward ladder) or a leaf in this
  batch (P1 consumes P0; P2 consumes P1; P3 and Φ consume P2); the riders PRD is named only as an
  additional consumer of a constant and predicate this PRD's own generator consumes.
- G3: no novel substrate on main; PRD 1's unlanded modules are hard prerequisites by task id, not
  assumed. Facts corrected while verifying: 8142 is `pending`, not in progress; `superseded_by` on
  main is the memory-store key, not the task key 8142 blesses; `fused-memory/tests/integration/`
  does not exist yet; `_apply_status_transition` takes the non-reentrant lock itself (so the CAS is
  a precondition inside it, not a wrapper around it); `_execute_combine` is lock-free (so the pack
  rewrite re-reads under the lock, the combine does not); the janitor's interval is captured at
  start (so the pack loop re-reads it); the watchdog ages from `eligible_since`, not
  `pending_since`; `effective_lock_depth` lives in `mcp_tools/scheduler_state.py`.
- G6: the only numeric claims are the re-run's measured stock figures (§2 table) quoted as context,
  not as pass bounds; the P0 fixture signal reproduces a partition count on a committed fixture,
  not a threshold. Rejection-style signals name the mechanism that fires: `PackExclusion` (P0,
  pure), `PackRefusal` from the live pre-check, from `rewrite_pack_survivor` under the lock and from
  `_apply_status_transition`'s `require_before` inside its critical section (P2), the parser
  degrade (P0), `DeferReason` (existing, wired at `invoke_with_cap_retry`), the `mode: off` early
  return (P1). Reader-shaped leaves (P3, Φ) render the no-producer state (decision 16). Row 14's
  end-to-end capability traces to P2's own prerequisites (8415, 8418, 8142, 8143) and never to a
  task downstream of it.

## 11. Critic fold-in (2026-11-04)

`task-shape-seats/PRD2-C.md` (six answer-changing, thirteen lesser) was applied in full with the
lead's choices where it left one. The in-lock CAS became a `require_before` precondition inside
`_apply_status_transition`'s own critical section with a thin wrapper (AC1; the wrapper shape would
have deadlocked on the non-reentrant lock); the survivor rewrite now re-reads under the lock
immediately before its write, the combine path stays lock-free, and the G7 "stale image never
applied" line is true for the survivor too (AC2); every yield, coverage, judge-call, cost and
drain figure was replaced by a re-run of a2 + c2 without b over today's stores, recorded with its
rule in `plans/followup-packing-sweep-prd.packs-a2c2.csv` (AC3: whole pool df 118 / 339 / 221, reify 134 / 385 / 251; the
earlier figures included b and summed to numbers no a2 + c2 partition could reproduce); the
starvation-watchdog row and series were dropped because the watchdog ages from `eligible_since`
(AC4); the rider reservation is owned here as a pure `rider_reserved` predicate keyed on
`pending_since` and `metadata.rider_ledger`, the report prints both coverage series, and decision 1
states that the reservation defers nearly half of df's with-files members and 54 of its 118 packs
from the first observe week (AC5); the judge moved to a new `pack_judge.py` with one
`render_pack_member`, and `task_curator.py` no longer grows (AC6). Lesser items: the image key is
no longer `updated_at` and a `stale` row is re-imaged without a judge call (b-2); the interval is
re-read per sleep (b-3); undo gives members a fresh `pending_since` by D3 (b-4); no in-pack edges,
plain ascending cancel order, `has_live_dependents` covers any live dependent or child (b-5);
promotion may cascade (b-6, decision 18); PRD 1 ε's waiver text gets the companion correction (b-7);
`role='curator_pack'` (b-8); reconciliation load justifies `max_packs_per_run` (b-9); the observe
period is ≥ 7 days and in practice until P2 lands (b-10); P2 adapts to 8143's threading (b-11);
first-pass cost stated once (b-12); the artifact triple replaces the corpus cite (b-13);
`effective_lock_depth` re-cited and §9's former item 5 closed (b-1).

Delta pass 1 (PRD2-C "Delta pass 1", same day) found one further answer-changing defect and two
lesser ones, all applied. AC7: P0's fixture named a pool dump that did not exist and could not be
derived from the pack CSV (hubs and the a2 union-find need every held member, packed or not), so
the full held member set behind the re-run was dumped read-only from today's stores as
`plans/followup-packing-sweep-prd.pack-pool-2026-11-04.json` — 2,517 records (df 1,405, reify
1,112) carrying task id, project, kind, parent, files, lock keys (equal to files at both depths),
`pending_since`, item count, a PRD 1 §4.2 band proxy, `has_live_dependents` and the v1 recipe, with
the rule, the read time, each project's lock depth and hub keys (8 / 16) in its header; the CSV was
moved beside it as `plans/followup-packing-sweep-prd.packs-a2c2.csv`; P0's row now cites both
plans/ files and copies them into `fused-memory/tests/fixtures/` itself, and its signal states the
exact eligibility the reproduction runs under (no decision-7 exclusions, as PRD2-P §0). Lesser:
the `rider_ledger.declines` read contract is stated once (decision 13, §4.1, §5: int, absent reads
0, the key and its writer are PRD 3's); tick step (2) now says `reimage` runs before the
no-re-judge filter, so `stale` enters that list only once re-imaged. The three files (PRD, pool
dump, CSV) are committed together by the lead; until then every leaf's PRD pointer is dead in a
worktree. Delta pass 2 (same day): c2's exact ordering and chunk-close rule, as the re-run script
applied them, are now pinned in §4.1 and the dump header (`c2_ordering`, `comparison`) so P0's
set-wise reproduction is meetable (AC8); decision 1 and §2 state that 118 / 134 are upper bounds
taken before decisions 7–8, with the live-store gap quantified (AC9); P0's row forces
`dependencies` empty and names every other forced field (lesser). Final round, after PRD 3
committed (`c393a67c54`, tasks 8438–8441): decision 7 names `rider_of` exactly as PRD 3 defines it
(a carried rider, never left set on a terminal or stalled host) and `exclusion()` reads
`rider_ledger.last_decline_reason` as a same-parent pack hint (`different_intent` → pack with
siblings; absent → no hint); §5 cites PRD 3 as committed with its task ids and records that γ 8440
will depend on P1 once filed.

## Decompose record (2026-11-04)

Decomposed against main `eb84ac9f41` by the PRD 2 decompose seat. The manifest and sidecar were committed
at `28ff3cc7de` and the stamped sidecar at `646bd7235b`, both as
`plans/followup-packing-sweep-prd.capability-manifest.{md,yaml}`, with `signal_leaves: [P3]`. There are 48
grep checks (no `script`, no `path`) and 8 manual ones; the local polarity lint returned 0 findings and
`shared/tests/test_capability_manifest.py` passed (551).

| label | id | prereqs (edges) | prio |
|---|---|---|---|
| P0 | 8451 | 8413 (PRD 1 α), 8414 (PRD 1 β) | high |
| P1 | 8452 | 8451, 8417 (PRD 1 ε) | high |
| P2 | 8453 | 8452, 8415 (PRD 1 γ), 8418 (PRD 1 ζ), 8142 and 8143 (cancel-with-children α, β; both `pending` on 2026-11-04) | high |
| P3 | 8454 | 8453 | medium |
| Φ | 8455 | 8452, 8454; deterministic, `always_escalates`, `execution_class: decision`, `milestone {delayed, 604800}` (clock starts when both are done) | medium |

Edges: 5 intra-batch, 7 out-of-batch. P0–P3 are `task_kind="normal"`; every task is pending. No task depends
on a blocked task (6707 is not an edge, §5). `commit_planning` stamped P0, P1, P2, P3 and Φ with no missing
labels or errors; `delivered_checks` copied: P0 19, P1 11, P2 15, P3 3, Φ 0. P0 and P2 carry
`g7_waivers: []`, as §6 states; the G7 re-walk found no unresolved hit, so no waiver.

**Companion corrections.** `8440 → 8452` wired (PRD 3 γ's pending edge, its §5 row and decompose record),
with a details note on 8440. A details note on 6322: the packer excludes `dedupe_skipped` tasks until its
sweep clears the marker, and it may reuse P1's loop host (no edge). 5197 is done (no write). 8417's waiver
rationale and PRD 1 decision 15 / row ε already read "the packing sweep and riders are the drains, bounded by
θ's report" (`dc1981e7a2` and the 8417 details note), so nothing was re-written. The cancel-with-children
§4.14 "pack executor" row is P2's edit.

**Resolved at decompose** (no contract change; carried into the descriptions):

1. G6, P0's fixture premise: an independent re-implementation of §4.1 over the committed pool dump
   reproduced all four CSV runs as sets of (generator, member set). It depends on three facts this PRD did
   not spell out, now in P0's description and the manifest: the v1 recipe regexes (only in the untracked
   seat notes and a `/tmp` script; quoted verbatim in 8451's details, and matching the dump's `recipe_v1`
   on 2,517 of 2,517 titles); the hub share is over the whole held pool, fileless members included; an
   absent `pending_since` (reify 7951) reads as oldest, never reserved.
2. Name correction: `fused-memory/tests/test_task_interceptor_curator.py` (§4.5 row 22, §6 P2) does not
   exist; the combine tests live in `fused-memory/tests/test_task_interceptor.py`, which is in P2's files.
3. INV-8 note on P1: member assembly is one bulk read per project, never a `get_task` per member.
4. P0 records `rider_ledger.last_decline_reason` on the proposal as a hint only; acting on it in generation
   is a v2 candidate (critic's final nit).
