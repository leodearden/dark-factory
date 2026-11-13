# Capability manifest — `plans/followup-packing-sweep-prd.md`

Per-leaf capability → evidence bindings (G3 + G6 mechanized). Machine-readable twin:
`plans/followup-packing-sweep-prd.capability-manifest.yaml` (task ids stamped by `commit_planning`).
Verified against main `eb84ac9f41` (2026-11-04): every `expect: present` pattern below is absent in its
`paths` today, and the polarity lint (`shared/delivered_check_polarity.py::lint_delivered_checks`) run
locally per leaf against `main` returned 0 findings. Every mechanical check is `kind: grep` on a symbol
or a content line; none is a `path` check or names a file the leaf creates as its pattern. Anchors are
symbols; nothing here is a line pin.

## Substrate (G3 re-check, symbol by symbol on `eb84ac9f41`)

Present and wired: `middleware/ticket_janitor.py::TicketJanitor.run_loop`,
`server/main.py::_build_ticket_janitor`, `config/schema.py::{TicketJanitorConfig, LinkHealConfig}`,
`config/reload.py::RELOADABLE_FIELDS`, `task_interceptor.py::TaskInterceptor.{_execute_combine,
_apply_status_transition, _write_lock, _append_combine_audit, _combine_audit_path, _advise_text_amendment,
_has_branch_base_sha}` and `COMBINE_REFUSED_CLAIMED` (the `details_with_files` build inside
`_execute_combine` is where P2's extraction starts), `curator_types.py::{is_combine_eligible_status,
clip_for_prompt, RewrittenTask, PoolEntry, POOL_SOURCES}`, `task_curator.py::{CURATOR_SINGLE_SPEC,
CURATOR_BATCH_SPEC, _CURATOR_HEURISTICS, _CURATOR_PROMPT_HARNESS_VERSION, TaskCurator._ledger_kwargs}`
and `invoke_with_cap_retry`, `curation_deferral.py::{DeferReason, DEDUPE_SKIPPED_KEY}`,
`mcp_tools/scheduler_state.py::effective_lock_depth`, `_folded_escalation.py::file_folded_escalation`,
`sqlite_task_backend.py::{stamp_pending_since, set_status_and_stamp_audit}`,
`reconciliation/targeted.py::TargetedReconciler._sweep_cancelled_descendants`,
`server/tools.py::{replay_dead_letters, rebuild_candidate_key_index}`,
`models/scope.py::build_known_projects_map`, `shared/locking.py::{files_to_modules,
strip_directory_locks}`, `shared/prompt_artifact.py::{PromptSpec, PromptArtifactStore}`,
`shared/task_metadata.py::{register_metadata_submodel, _BLESSED_METADATA_KEYS}`,
`orchestrator/eligibility_clock.py::age_secs`, `scheduler.py::Scheduler._update_age_anchors`.

Absent, produced upstream (all pending, wired as out-of-batch edges): `shared/declared_scope.py` and the
reader allowlist test `shared/tests/test_declared_scope_readers.py` (PRD 1 α 8413 → P0);
`shared/followup_items.py` with `MAX_FOLLOWUP_ITEMS`, `FollowupItem`, `merge_items` (β 8414 → P0); the
item-preserving combine and `followup_hold.py` with `is_holder` (γ 8415 → P2); the hold record, holder and
backfill (ε 8417 → P1); `promote(by='pack')` and its `by='dependent'` cascade (ζ 8418 → P2);
`superseded_by` blessing (cancel-with-children α 8142 → P2) and its atomic stamp with the cancel through
`_apply_status_transition` (β 8143 → P2). Both 8142 and 8143 read `pending` on 2026-11-04 (`get_task`).
Absent as the PRD says (this batch creates them): `followup_packing.py`, `pack_judge.py`,
`pack_ledger.py`, `pack_sweep.py`, `pack_executor.py`, `scripts/pack_sweep_report.py`,
`fused-memory/tests/integration/`, every `pack_*` / `Pack*` symbol, `RIDER_RESERVE_DAYS`.

Name correction (no contract change; carried into P2's description and file list):
`fused-memory/tests/test_task_interceptor_curator.py` (PRD §4.5 row 22, §6 P2) does not exist. The
filing-time combine tests live in `fused-memory/tests/test_task_interceptor.py` (also
`test_ticket_worker.py`, `test_task_curator.py`); P2 keeps those green through the extracted half.

## G6 — the P0 fixture premise, re-run at decompose

P0's signal reproduces `plans/followup-packing-sweep-prd.packs-a2c2.csv` from
`plans/followup-packing-sweep-prd.pack-pool-2026-11-04.json` (both committed with the PRD at
`eb84ac9f41`). An independent re-implementation of PRD §4.1 over the dump, with the dump header's
`c2_ordering` and forced fields, reproduced all four runs as sets of (generator, member set): df whole
pool 118 (a2 77, c2 41), df minus reservation 64 (34, 30), reify whole pool 134 (89, 45), reify minus
reservation 112 (76, 36). Three facts the reproduction depends on, which the PRD does not spell out and
which P0's description now states:

1. **Recipes.** Every dump record carries `recipe_v1`; the re-run's v1 regexes (loose v0 table, first
   match wins, with `cite_doc_hygiene` replaced by the tightened v1 regex and a loose-only cite match
   reading as no recipe) agree with it on 2,517 of 2,517 titles. The regex table lives only in the
   untracked seat notes and a `/tmp` script, so P0's details carry it verbatim and P0's test asserts
   `title_recipe(title) == recipe_v1` on every dump record.
2. **Hubs.** The header's 8 / 16 hub keys come from a > 2% share of the WHOLE held pool, fileless members
   included (denominators 1,405 / 1,112); the with-files denominator gives 13 / 19 and does not reproduce.
   The minus-reservation runs use the same whole-pool hubs.
3. **Absent `pending_since`.** One reify record (7951) has none. The CSV reproduces only when it reads
   as NOT reserved (943 eligible, not 942), so an absent `pending_since` reads as oldest.

## Bindings

| leaf | capability | binding | verdict |
|---|---|---|---|
| P0 | `candidate_packs`, `hub_keys`, `survivor_of`, `title_recipe`, `rider_reserved`, `PackExclusion` | new `middleware/followup_packing.py`; grep each definition there | PASS |
| P0 | `RIDER_RESERVE_DAYS = 14`, `PACK_MAX_UNION_KEYS = 15`, `MAX_FOLLOWUP_ITEMS` imported | grep each in `followup_packing.py`; `MAX_FOLLOWUP_ITEMS` producer is 8414, upstream | PASS |
| P0 | `CURATOR_PACK_SPEC`, `curator-pack-v1`, `_CURATOR_HEURISTICS` by reference, `parse_pack_verdict` (rejection-style degrade), `render_pack_member`, `judge_pack` | new `middleware/pack_judge.py`; grep each | PASS |
| P0 | pool and expected-output fixtures copied into `fused-memory/tests/fixtures/` | grep a content line in each copy (`"c2_ordering"`, `^dark_factory,minus_reservation,`); the sources are the two committed `plans/` files | PASS |
| P0 | fixture reproduction | grep the fixture name in `test_followup_packing.py`; set equality is manual (reproduced at decompose, above) | PASS |
| P0 | allowlist gains `followup_packing` | grep in `shared/tests/test_declared_scope_readers.py` (created by 8413, upstream) | PASS |
| P0 | no curator or interceptor import | manual (an absent-grep on an uncreated file is vacuous) | PASS |
| P1 | `PackLedger`, `producer_has_written`, `reimage` | new `middleware/pack_ledger.py`; grep each | PASS |
| P1 | `PackSweep`, start-log line | new `middleware/pack_sweep.py`; grep `class PackSweep[(:]` and `Pack sweep: mode=` | PASS |
| P1 | `PackSweepConfig`, hot-reloadable | grep in `config/schema.py`; `curator\.pack_sweep\.` in `config/reload.py` | PASS |
| P1 | started at startup; observe shipped | grep `PackSweep` in `server/main.py`; `pack_sweep:` in `fused-memory/config/config.yaml` | PASS |
| P1 | allowlist gains `pack_sweep`; rows 1–5, 11–13, 21 | grep in the allowlist test; `def test_` in new `test_pack_sweep.py` | PASS |
| P1 | observe rows in production | manual; read by P3's report | PASS |
| P2 | `apply_pack`, wired into the enforce arm | grep in new `pack_executor.py` and in `pack_sweep.py` | PASS |
| P2 | `require_before` precondition (rejection-style), `cancel_superseded_if_pending` | grep in `task_interceptor.py`; on top of 8143 (upstream) | PASS |
| P2 | `_write_combined_rewrite` extracted and called by `_execute_combine` | grep the def and `await self\._write_combined_rewrite\(` | PASS |
| P2 | `rewrite_pack_survivor` (rejection-style refusals under the lock), `undo_pack` method and MCP tool | grep in `task_interceptor.py` and `server/tools.py` | PASS |
| P2 | `rehold` | grep in `followup_hold.py` (created by 8415, extended by 8417/8418, upstream) | PASS |
| P2 | `basis` gains `'pack'`; `pack_record` registered and documented | grep in `shared/declared_scope.py` (8413 upstream), `task_metadata.py`, `docs/task-authoring.md` | PASS |
| P2 | cancel-with-children §4.14 "pack executor" row | grep `[Pp]ack executor` in `plans/cancel-with-children-prd.md` | PASS |
| P2 | rows 14–19, 15b, 16b, 22; claim races refused inside the lock | `def test_` in new `test_pack_executor.py`; the race property is manual | PASS |
| P3 | report renders `producer has not written` | grep in new `scripts/pack_sweep_report.py` | PASS |
| P3 | report tests, boundary suite | `def test_` in `tests/scripts/test_pack_sweep_report.py` and new `fused-memory/tests/integration/test_pack_sweep.py` | PASS |
| P3 | §4.5 rows 1–22 green; report over the live ledger | manual | PASS |
| Φ | Leo's ruling on one L2; enforce rows after an enforce ruling | manual | PASS |

56 capabilities: 48 `kind: grep`, 8 manual. No binding resolves to declared-only, test-only,
producer-downstream, producer-absent, producer-extent-short, fixture-ERROR or rejection-absent; every
rejection-style capability (`PackExclusion`, the parser degrade, `PackRefusal` from the pre-check, from
`rewrite_pack_survivor` and from `require_before`) is created in the leaf whose signal asserts it.
DAG: P0→8413, 8414; P1→P0, 8417; P2→P1, 8415, 8418, 8142, 8143; P3→P2; Φ→P1, P3. PRD 3 γ 8440→P1
(companion correction). `signal_leaves: [P3]`.
