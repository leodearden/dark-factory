# Capability manifest — plans/dashboard-alignment-prd.md (stream M3)

Per-leaf capability→evidence bindings (mechanized G3+G6). All greps run
against main @ e19aeea088-lineage, 2026-07-06. No FAIL bindings.

## α — OutcomeKind in merge_types.py + typed emit chokepoint

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `_emit_merge_attempt(event_store, task_id, outcome: str, ...)` exists as the single emit chokepoint | grep: orchestrator/src/orchestrator/merge_queue.py:1352 (def) | PASS wired |
| The 21 outcome strings are exactly the live call-site set | balanced-paren extraction over merge_queue.py (749, 1466, 1507, 2552, 2580, 2638, 2819, 2853, 2871, 2890, 2958, 2974, 3092, 7271, 8911, 8941, 8984, 9004), merge_gates.py (421, 618-via-GateVerdict.emit_subtype {387: post_merge_equivalence_failed, 462: post_merge_pyright_broken}, 644), workflow.py (5364, 5376) | PASS enumerated |
| StrEnum precedent in merge_types.py (payload str-compat) | grep: merge_types.py:890 `class InflightStatus(StrEnum)` — same file, same serialization contract | PASS wired |
| Event payload unchanged under member emission | StrEnum member str-identity (`str(E.X) == 'x'`, json via `data: dict = {'outcome': outcome}` at merge_queue.py:1381) + α's payload-identity test | PASS |
| Non-terminal set {cas_retry, gate_retry, post_merge_generation_chained, plan_files_narrowed} | traced: cas_retry merge_queue.py:9004 (retry continues), gate_retry :8941, post_merge_generation_chained merge_gates.py:421 (enqueue_merge_request already called inside `_maybe_auto_chain_generation` merge_queue.py:1741 before the emit), plan_files_narrowed workflow.py:5376 (submission proceeds to enqueue in same method) | PASS |
| Pyright pre-commit enforces the typed param | repo pre-commit runs pyright 3x (project convention); str→OutcomeKind narrowing is a basic-mode error | PASS |

## β — Fail-safe inversion of dashboard classification

| Capability asserted | Evidence | Verdict |
|---|---|---|
| The two divergent lists exist to delete | grep: dashboard/src/dashboard/data/merge_queue.py:51 `_CANONICAL_OUTCOMES`, :53 `_TERMINAL_MERGE_OUTCOMES` (8 members, frozen at commit 8553b388ad) | PASS |
| Drift premise (G6): terminal strings missing from the frozen list | set-diff: {merge_failed, verify_failed, advance_failed, train_incomplete, train_rebase_conflict, train_partial_flip, main_health_red, post_merge_equivalence_failed, post_merge_pyright_broken, plan_files_not_touched, plan_files_narrowed} ∉ _TERMINAL_MERGE_OUTCOMES; consumer at :783 keeps such tasks "active" for the 30-min TTL | PASS (premise true) |
| `active_queued_merges` consumer path | grep: dashboard merge_queue.py:717 (def), :783 (`outcome in _TERMINAL_MERGE_OUTCOMES`), :952 (caller in build_per_project_merge_queue) | PASS wired |
| Panel impact is real (not dead code) | task 1606 (done) made live get_merge_queue primary and KEPT the event-derived list as fallback for unreachable orchestrators; `outcome_distribution` (:281-288) uses _CANONICAL_OUTCOMES on every request | PASS (fallback + doughnut both live) |
| Producer of the ACTIVE_ONLY mirror is upstream | producer: task α (OutcomeKind non-terminal members + frozen-contract test), wired as dep β→α | PASS producer-upstream |

## γ — resolve_now + costs now-threading

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Latent bare clock reads exist | grep: costs.py:33 (`_cutoff` → `datetime.now(UTC)` per call), costs.py:440 (`get_cost_trend`) | PASS (premise true) |
| Per-DB fan-outs re-derive now | grep: costs.py:630-836 `aggregate_cost_*` gather per-DB calls with no `now` param; app.py:714-719 gathers 6 aggregates concurrently | PASS |
| The exemplar pattern exists | grep: dashboard merge_queue.py:100 `effective_now = now if now is not None else datetime.now(UTC)`; :908 `now: datetime` threaded (task 692 / 726 lineage) | PASS wired |
| No open duplicate task | search_tasks: 726 (done, merge_queue only), 317 (done, intra-call fix only), none open for costs/burndown | PASS |

## δ — burndown now-threading + clock-discipline guard

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Latent bare clock reads exist | grep: burndown.py:509 (`get_burndown_series`), :108/:220 (writer paths — exemption-tagged, not threaded) | PASS (premise true) |
| Per-DB fan-out re-derives now | grep: burndown.py:299-301 `aggregate_burndown_series` gathers `get_burndown_series(db, ...)` per DB, no now param; caller app.py:1301 | PASS |
| Guard is enforceable by grep | all `datetime.now(` sites in dashboard/src/dashboard/data/*.py enumerated this session (burndown ×3, costs ×2, merge_queue fallback expressions); post-conversion residue = resolve_now def + `# clock-exempt:` tags — the test's negative fixture proves it fires | PASS |
| Producers upstream | γ (resolve_now helper), β (same-file edit serialization on dashboard merge_queue.py) — wired as deps | PASS producer-upstream |

## ε1 — mcp_fanout helper + memory/tasks conversion

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Duplicated failover loops exist | grep `for url in config.fused_memory_urls`: memory.py:204/232/268, tasks.py:149/195/235, scheduler.py:625, metrics.py:105, app.py:820/1043 | PASS (premise true; tasks.py:36 comment even names the missing shared helper) |
| Duplicated TTL caches exist | grep: tasks.py:54 `_FETCH_TASKS_TTL_SECONDS`/`_fetch_tasks_cache`, scheduler.py:74/82 `_SCHEDULER_TTL_SECONDS`/`_scheduler_cache`(+refresh lock), dashboard merge_queue.py `_task_titles_cache`, app.py `_task_cards_cache` | PASS |
| `invalidate_session` substrate for the helper | grep: memory.py:163 (def), used in memory.py ×4 / scheduler.py ×2 | PASS wired |
| Existing tests observe the converted panels | dashboard/tests/test_memory.py, test_active_tasks.py present | PASS |

## ε2 — remaining conversions (scheduler/metrics/app/merge_queue)

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Target loops/caches | scheduler.py:625 `_one_project` (paired-calls-per-URL shape — helper's callable-per-URL signature covers it), metrics.py:105, app.py:820/1043, merge_queue.py `_task_titles_cache` | PASS |
| Producer upstream | ε1 (helper module) + δ (same-file serialization on app.py / dashboard merge_queue.py) — wired as deps | PASS producer-upstream |
| Non-targets documented (not silently skipped) | get_queue_stats/get_wal_status (memory.py:223/254, sum-across-all) and merge_halt.py (concurrent probe-all) are different idioms — PRD Out-of-scope names them | PASS |

## ζ — format-coupling doc block

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Re-derived formats exist as claimed | grep: dashboard orchestrator.py:140 `'orchestrator run' not in line` (ps-scan), :155-156 `--prd/--config` regexes, :181-227 `.task/` layout (metadata.json, plan.json steps/files, iterations.jsonl, reviews/*.json verdict) | PASS (premise true) |
| Source-of-truth targets exist | orchestrator/src/orchestrator/cli.py `run` click command (:168-183 options); orchestrator/src/orchestrator/artifacts.py (25 plan.json/iterations.jsonl refs) | PASS |
| No competing ownership | `.task/` relocation owner = W11 per program seam table; ζ moves NO derivation logic (document-only per brief) | PASS |
| Import unification correctly rejected | dashboard/pyproject.toml has no orchestrator dependency (escalation + dark-factory-shared only) | PASS |
