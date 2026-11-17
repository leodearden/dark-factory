# S12 — Reconciliation run lifecycle and stage interface (PRD)

**Status:** draft, authored 2026-11-11 by the S12 author seat of the `/review-all` remediation program
(`plans/review-all-program-dark_factory-2026-10-10.md`, stream S12, wave 3, size L, risk medium, ruling R12).
B+H: a contract section (C1–C7) and a two-way boundary-test sketch. Seventeen tasks: sixteen leaves (the
fold added γ0) and the integration gate ω. Nothing is held. No invariant is promoted (R22 names none for
S12). Folded 2026-11-11 against the critic pass. Ruling fold 2026-11-12: every open question was ruled at
the wave-3 sitting (§Rulings) and the text is authored under the rulings; no question is open.
Fold-verified 2026-11-12 (main `0f21f5e457`). Aligned 2026-11-13 with the sibling drafts and the lead's rulings
after the wave-3 fold-verifiers (test-file edges unwired, S12-2's three readings confirmed, the
decision-partition invariant cited by its slug).

**Code anchors** verified against main `1b106418da` (2026-11-12). Main moves fast — cite-by-symbol;
re-locate lines at implementation time. Re-checked at fold-verify against main `0f21f5e457`
(2026-11-12): none of the files this plan cites changed between `1b106418da` and `0f21f5e457` (40 commits),
so every count below holds there. The run pinned `6551b75aba` (2026-10-08); the author's anchor was
`1f240e6da2` (2026-11-10). Between `1f240e6da2` and `1b106418da` (743 commits) the files this plan cites
that changed are `fm/reconciliation/harness.py` (+15/−16, task 6158: the registry derivation and a
read-only `known_projects` property), `prompts/__init__.py` (+32/−5, task 9317), `prompts/stage2.py` (+3/−1,
task 4867), `stages/task_knowledge_sync.py` (+9/−9), `server/main.py` (+64/−46: task 4867's
unknown-argument guard, task 6158), `server/tools.py` (+38/−14: task 6621's moves, tasks 4695 and 6956),
`middleware/task_interceptor.py` (+155/−279: task 6621's moves, task 4695) and the test modules `test_harness.py`, `test_stages.py`,
`test_task_interceptor.py` (`git diff --stat 1f240e6da2..1b106418da`); each symbol this plan relies on in
them was re-read at `1b106418da`. Every count is labelled `measured at 1b106418da`: counts resting on a
changed file were re-measured; every other cited file is byte-identical between the two shas, so its
counts hold. `fm/` abbreviates `fused-memory/src/fused_memory/` in prose; Files lists spell full
repo-relative paths. Under the lead's ruling S12/OQ-12 the S19 moves land first, so Files lists name
`journal.py`, `event_buffer.py` and `recon_self_model.py` at their `fm/recon_state/` paths; prose cites
them by module and symbol, which the moves do not change.

## Goal

Reconciliation gets one run state machine, one typed run record and one stage interface. Today a
reconciliation run's lifecycle is written twice, its record is a heterogeneous dict, its stages receive their
inputs through attribute assignment and hand their outputs back through instance attributes, and a stage's
identity is a string the prompt asks the LLM to echo.

What a user observes when this lands:

- `fused-memory.service`'s journal (`data/reconciliation/reconciliation.db`, table `runs`) shows every run
  kind — a full cycle, a backlog chunk, a remediation pass — ended by the same rules. A cancelled run of any
  kind is `failed` with its own drained events back in the buffer. A run whose `complete_run('completed')`
  committed stays `completed` even when its post-completion tail fails, remediation passes included (today a
  failing judge spawn or persistence gate relabels a completed remediation run `failed` and files
  `recon_integrity_issue`).
- No backlog event is lost to a shutdown between a drain and the run row: the run row exists before any event
  is drained, so the stale-run reaper restores it.
- Every `write_ops` row a recon stage causes carries `agent_id='recon-stage-<stage>'` and
  `causation_id=<run_id>` because the harness bound them on the stage's MCP connection, not because the LLM
  remembered a sentence in its prompt; that includes the creation-journal rows of the tasks Stage 2 files.
  The stage prompt now says the opposite, in so many words: "The connection already identifies you and this
  run. Do not pass `agent_id` or `_causation_id`." A stage write that passes a different `agent_id` or
  `_causation_id` anyway is refused with a structured error naming both values, writes nothing, and leaves
  one log line; a matching or missing value lands, attributed to the header (`search`, whose `agent_id`
  filters by authoring agent rather than naming the caller, is outside the rule). `CLAUDE.md`, the
  `/memory` command, `docs/task-authoring.md` and the fused-memory server's own MCP instructions
  (`server/tools.py::FUSED_MEMORY_INSTRUCTIONS`, which every client, stages included, receives) carry the
  one-line exception.
  The stage's recon-report connection carries the same two headers, for the recon-report server to read
  under the same rule (S13).
- `python -c 'from fused_memory.reconciliation.journal import ...; get_run(...)'` (any reader) gets a
  `ReconciliationRun` whose `stage_reports` is keyed by `StageId` and holds only `StageReport`s, with the
  failure in `run.error`. No reader branches on `isinstance(report, dict)`.
- A Stage-2 briefing-refresh task appears on the curator's ticket path with a `task_created` event, like
  every other task recon files; a targeted-reconciler write the interceptor refuses is journaled as a
  refusal, and each degraded arm of `_on_task_done` leaves a `run_actions` row. The one deliberate exception,
  the auto-consolidation hazard gate filed on the raw backend (`plans/memory-auto-consolidation-prd.md` D8), is
  named in the guard with a ratified INV-12 disposition.
- The `runs` table keeps `events_processed` for every run kind, now written once the drain has counted the
  events.
- Adding a run kind, a stage input, a stage output statistic, an "N bad cycles in a row" alarm, or a stage
  capability is one edit in one module. The next agent does not have to mirror a change across two drivers,
  a reset, a stash and a read-back, or a second streak quartet.

## Background

Run `review-all-dark_factory-20261008` (record `plans/review-all-dark_factory-20261008.json`) found the
reconciliation core carries these costs (heuristics cited by name from `docs/code-quality.md`):
**SPOT** and *simple control flows* — the run lifecycle exists twice (`run_full_cycle`,
`_run_remediation_pass`) and its fixes arrive in pairs; *structured data instead of meaningful strings* —
`ReconciliationRun.stage_reports` mixes stage reports with `_error`/`_resume` records and synthetic dicts,
stage identity is a prompt-echoed `recon-stage-<id>` string, and the cycle-summary write state is a set of
string-keyed stats flags; *prefer stateless interactions between modules* and *minimum data access scopes
and lifetimes* — stages receive per-cycle inputs by post-construction assignment and route per-run outputs
through instance attributes; *clear invariants uniformly enforced* — recon task writes reach the
TaskInterceptor only where a caller remembered to route them; *files make internal sense in isolation* and
*no file too large* — `harness.py` and `prompts/__init__.py`.

Ruling R12 (2026-10-10) chose option 1: one run driver parameterised by `RunKind(full | remediation |
backlog_chunk)` with scope-freshness as a pre-step, used by `run_full_cycle`, `_run_remediation_pass` and
`BacklogIterator`, and decided the rest "as proposed". This PRD implements R12 without relitigating it. R12
leaves one decision to this PRD — the CancelledError/resume asymmetry as one rule — which D2 makes. R22 names
no invariant for S12; R23 names no weakened S12 key.

What changed since the pin and matters here: task 5514 retired adopt-and-resume (no code writes
`RunStatus.interrupted`; a cancelled `run_full_cycle` ends `failed`), task 7230 added a durable-completion
guard to `run_full_cycle`'s failure arms only, task 6453 made every drain run-attributed, task 6120 broke
`MemoryConsolidator.run` into named steps and moved the Stage-1 flag chain to a declared filter table, and
task 6463 lets the harness take its escalation queue at construction. Since the author's anchor: task 4867
(done, merged 2026-11-11, `daff96ef3c`) made fused-memory refuse a tool argument its tool does not declare
(`server/unknown_argument_guard.py`, one log line per refusal, the refusal built by
`shared/src/shared/mcp_unknown_arguments.py::unknown_argument_refusal`), the precedent for C3's
self-explaining refusal; task 6158 gave `ReconciliationHarness` a read-only `known_projects` property.

## Rulings (sitting of 2026-11-12)

Every open question of this stream was ruled on 2026-11-12 (`digests/wave3-rulings-applied.md` §S12 and its
cross-stream block; Leo's answers `digests/wave3-leo-answers-2026-11-12.md`; for OQ-3 the measured "Effect on
the plan if adopted" in `digests/discussion/S12.md` §S12-2 is part of the ruling; lead rulings
`digests/wave3-lead-rulings.md`, later blocks winning). The text of this PRD is authored under them. No
ruling below amends a program ruling: OQ-3 sharpens R12's "StageId in the run context rather than a
prompt-echoed agent_id string" (an echoed value is now refused, not merely unneeded). OQ-3's run-id clause and
its cross-stream rule supersede the withdrawn lead ruling S12/OQ-3 ("header wins, one override event") and
the lead's 2026-11-11 alignment line on recon-report identity.

| Id (sitting key) | Ruling | Ruled by |
|---|---|---|
| OQ-1 (S12-1) | As recommended, option 3: a streak alarm files on every bad cycle whose shared-counter count is at or over its threshold; repeats fold onto the open record; a resolved-but-persisting condition files again on the first bad cycle after the 24-hour recently-resolved window (D8, C5, boundary row 12). | Leo |
| OQ-2 | Option 4 (duplicate of S15/OQ-9, which Leo ruled (a)): the reconciler factory is a provider member of task 9280's frozen `InterceptorCollaborators` bundle (C6); S15 η → S12 κ1. | the lead |
| OQ-3 (S12-2) | Option B: the server refuses a stage write whose `agent_id` differs from the stage-identity header, with a structured error naming both values and saying the connection already identifies the stage, and one log line per refusal; a matching or missing value lands, attributed to the header. Plus Leo: sweep every prompt and skill a stage reads, in δ, in the same commit as the refusal (the explicit "do not pass" line in the stage prompt; `prompts/stage1.py`, `prompts/stage2.py`, `recon_self_model.py`, `CLAUDE.md` §"Write-Tagging Convention", `.claude/commands/memory.md`; the fold adds `docs/task-authoring.md` §"Write-tagging convention", which states the same "every write should carry `agent_id`" rule with the same `"reconciliation-stage-1"` example and which `CLAUDE.md` tells every session to consult; confirmed by the lead); task filing reads the run id from the header (δ gains `task_interceptor.py` and the edge onto 9282); `recon_stage_identity_override` is gone everywhere; no new task. Plus the lead (S12.md residual R-1 = A): the refusal also covers a `_causation_id` that differs from the run-id header. The fold applies the one rule to the three read tools whose `agent_id` names the reader (`get_entity`, `get_entity_by_uuid`, `get_episodes`; D7; confirmed by the lead); `search`, whose `agent_id` is an authoring-agent filter, is outside the rule (fold-verify correction, confirmed by the lead). The fold-verify grep at `0f21f5e457` adds one more sweep site, `server/tools.py::FUSED_MEMORY_INSTRUCTIONS` (its "Include agent_id for attribution (e.g. \"claude-interactive\", …)" convention reaches every stage; confirmed by the lead). The lead confirmed all three readings in its rulings after the wave-3 fold-verifiers, and ruled that S15's tool-family moves carry `search`'s exclusion through verbatim. Leo's question ("the model doesn't send an ID") is answered in S12.md §S12-2 and relayed by the lead; it changes nothing further here. | Leo (run-id clause: the lead) |
| OQ-4 | Option 1 (flagged): the briefing-refresh writer files through `submit_task` and the curator and records ticket ids (D10, κ2). | the lead |
| OQ-5 | Option 1: module names as proposed; C1 and C2 share one `run_context.py`. | the lead |
| OQ-6 | Option 1: this plan goes after wave 2 on every shared production file; `targeted.py` is left to file locks against 9285 (D12). | the lead |
| OQ-7 | (a): only ε waits on S1 ι1; β1 and S1 ι1 are unordered; S1 ι2 drops the growth alarm, which ε moves onto S1's registry; S1 ω waits on S12 ε; the growth alarm's registry entry files at level 1 as today (`FilingPolicy.open_l1`). | the lead |
| OQ-8 | (a) (duplicate of S13/OQ-6): S13 ι, the verbatim split of `stages/task_knowledge_sync.py`, lands first; S12 γ0, γ1, ε and κ2 wait on it; S13 θ3 and S13 ε wait on S12 ι2; S13 ι takes no edge on any S12 task. | the lead |
| OQ-9 | (a) (flagged): the auto-consolidation hazard gate keeps writing on the raw task backend (`plans/memory-auto-consolidation-prd.md` D8); the guard's one exemption is ratified as INV-12 row `inv12-recon-hazard-gate-raw-backend` (D10, κ2). | the lead |
| OQ-10 | (a): δ puts the two headers on the stage's recon-report connection; S13's recon-report leaf (S13 ξ) makes the recon-report server read them. Under OQ-3's cross-stream rule S13 ξ refuses a mismatching run id instead of overriding it and removes the prompt's "echo run_id" line in the same change. | the lead (refusal clause: Leo's S12-2 via the cross-stream rule) |
| OQ-11 | (a1) plus the edge the server-boundary plan needs: S12 δ waits on S15 α only; S15 δ and S15 ε1 wait on S12 δ. | the lead |
| OQ-12 | (a): S19's moves of the journal, event buffer and queue, ledger and self-model land first; this plan names the `recon_state` paths. | the lead |

Two cross-stream rules from the same sitting bind this plan. Leo's principle "one decision partition per
decision area", the design invariant `one-decision-partition-per-area` (INV-23, minted last of the six wave-3
invariants by the scheduler plan's task; checks anchor on the slug, never the number), is applied in D13: every table and rule this plan builds is decided by code, and the only LLM judgement is
named there with its path to a human. The hermetic-evals rule is not engaged: this plan adds or moves no
store, sink, filer, queue or archive that an eval cell can reach (D13).

## Findings disposition (re-verified at `1b106418da`)

Every key of the stream appears exactly once. Contract §8 step 1 (`find_tasks_by_metadata(key=
"x_finding_key", value=<key>)`) returned no task for any of the 21 keys (re-run 2026-11-12; 322 tasks carry
the key). Step 2 (`search_tasks` paraphrases, ≥ 0.6, confirmed by `get_task`) found the live tasks in
§Live tasks; the 2026-11-12 re-run for δ's widened scope also surfaced task 6895 (score 0.57, below the 0.6 bar,
read with `get_task` because it names the same `_causation_id` convention), which the OQ-3 ruling supersedes
(§Live tasks).

| Key | Sev | Disposition | Note (measured at `1b106418da`) |
|---|---|---|---|
| `fk-8621784d79d3` two hand-mirrored run drivers | high | leaf β2 (β1 builds the driver) | `run_full_cycle` (300 lines, plus 8 helper methods task 7230 extracted) and `_run_remediation_pass` (842 lines) each still own `journal.start_run`, a stage loop with the pre-Stage-3 `_flush_cycle_summaries`, `complete_run`, `verify_and_rewrite_stats`, `update_run_stage_reports`, `_spawn_judge`, a three-arm except ladder and a `finally` that flushes, persists and GCs |
| `fk-2571f7726184` the two copies diverged | medium | leaf β2 | The resume half is gone (task 5514: both cancel arms end `failed`). New divergence: only `run_full_cycle`'s arms go through `_end_run_unless_completed` (task 7230); `_run_remediation_pass` calls `complete_run('completed')` and then runs verify, persist, judge and the persistence gate inside the same `try`, so a tail failure relabels a durably completed remediation run `failed`. The `AllAccountsCappedException` `_error` dict literal is still byte-identical in both |
| `fk-377a7036ac7f` `stage_reports` is a heterogeneous bag | medium | leaf α | Type `dict[str, StageReport \| dict]` (`models/reconciliation.py::ReconciliationRun`). `'_error'`: 7 writes, 1 pop, 1 read in `harness.py`; `'_resume'`: 1 legacy reader (`journal.py::ReconciliationJournal.get_run_stage_execution`). Shape-branching readers: 4× `isinstance(s3_report, dict)` in `harness.py`, `journal.py::_row_to_run` (`'stage' in v`), `judge.py::Judge._build_review_prompt`, `stats_verifier.py::_apply_observed`, `task_count_snapshot_cadence.py::extract_snapshot_written`, `task_knowledge_sync.py::_extract_growth_sweep_failed`, plus 3 new since the pin (`citation_repair.py`, `cross_run_citation_scan.py`, `episode_window.py`). Stage-key spelling in `harness.py`: 7 literals vs 4 enum-derived (rule: `grep -n stage_reports` classified by key expression). R12's resume attribute is moot: task 5514 retired resume; legacy `_resume` rows decode into `legacy_meta`, read only by `get_run_stage_execution` |
| `fk-f23c6e137ce5` cycle-summary write state as string-keyed stats flags | medium | leaf ζ | `_cycle_summary_ledger_write_missing` still parses `f'{prefix}_cycle_summary_*'` keys with `== 0` / `is True`; the three stat names appear on 29 lines of `harness.py`, 3 of `memory_consolidator.py`, 1 of `task_knowledge_sync.py`, 1 of `summary_pool.py`, and in 4 test files |
| `fk-0cce86f5dbfe` `assemble_payload` cannot carry a run or return stats | medium | leaf γ1 (γ0 moves the test call sites first) | `BaseStage.assemble_payload(events, watermark, prior_reports) -> str` unchanged. `TaskKnowledgeSync`: 6 stashes, 24 `self.<attr>` references (`_current_run_id` 10, the other five 14). `MemoryConsolidator`: 4 stashes, 23 references. Tests assign `_current_run_id` at 6 sites in 3 files |
| `fk-a3598aec7dca` stage identity is a prompt-echoed string | high | leaf δ (the four-homes half: S14 μ, task 6721) | `BaseStage.run`'s `recon_context` still instructs ``agent_id: "recon-stage-<id>"`` and `"_causation_id"`; `prompts/stage1.py` shows `propose_consolidation(…, agent_id='recon-stage-memory_consolidator')`; `prompts/stage2.py` asks for `metadata={…, "_causation_id": <run_id>}` on `submit_task`, and `recon_self_model.py`'s task-creation accounting text says outcomes are journaled "ONLY when that `submit_task` call carried" it; `stats_verifier.py` keys on `f'recon-stage-{stage_id}'`; 44 lines holding the literal `'recon-stage-'` in 16 `fm/` modules (`server/tools.py` 14; rule: `git grep -c "'recon-stage-'"` per file). Today an explicit `agent_id` argument always wins (`server/tools.py::_resolve_identity`); every one of the 28 MCP tools that declares `agent_id` resolves it there (rule: AST over `server/tools.py`, public functions with an `agent_id` parameter that call `_resolve_identity`). Task 6721 (S14 μ, pending, high) parses the identity once; it does not change where the identity comes from |
| `fk-d2c7f1b902ac` fused-memory URL from a client `.mcp.json` | medium | leaf δ | `stages/base.py::_find_fused_memory_server` still tries `Path('/home/leo/src/dark-factory/.mcp.json')`, then `cwd/.mcp.json`, then a stdio fallback, although `server/main.py::_require_http_transport_for_reconciliation` guarantees http |
| `fk-63b65973bd33` per-cycle inputs by attribute assignment | medium | leaf γ2 (γ0 moves the test construction first) | Now 13 "set by harness" class attributes on `MemoryConsolidator` (11 at the pin; `filtered_task_tree_fetched_at` and `overreach_flags` added), 3 on `TaskKnowledgeSync`, `filtered_task_tree` on `IntegrityCheck`, and `known_projects`/`_escalation_url`/`_escalation_queue` on `BaseStage` (via `_propagate_escalation_queue`). 19 test files assign these attributes; 34 test modules reach a stage by direct construction, `await stage.run(`, harness-set attribute writes, `._escalation_(queue\|url) =` writes or `.assemble_payload(` calls (the critic's union rule, `scratch/S12-critic/stage_test_union.txt`) |
| `fk-37356472614a` `MemoryConsolidator.run` 1,142 lines, eleven hand-sequenced filters | medium | `fixed:f600f869db` | Task 6120 (`f600f869db`, with `c7579e6d1c` and `747c90926c`): `MemoryConsolidator.run` is 48 lines of named steps, and the post-flight filters are the declared tables `stage1_flag_chain.py::RESOLVING_FILTERS_BEFORE_SUPPRESSION` / `RESOLVING_FILTERS_AFTER_DEDUP` run by `run_stage1_flag_chain`. The post-flight axis has one mechanism. No leaf |
| `fk-90e488a174b7` one streak policy implemented twice | medium | leaf ε | Both quartets present (`task_count_snapshot_cadence.py` extract/compute/evaluate + `ReconciliationHarness._maybe_escalate_stale_task_count_snapshot`; `task_knowledge_sync.py::_extract_growth_sweep_failed`, `_compute_growth_sweep_failure_streak`, `_evaluate_growth_sweep_escalation`, `_maybe_escalate_growth_sweep_failures`); two filing paths: `ReconciliationHarness._maybe_escalate_stale_task_count_snapshot` files `recon_stale_task_count_snapshot` at info severity through `_escalate`; `_maybe_escalate_growth_sweep_failures` files `Escalation(level=1, severity='blocking')` on the synthetic key `recon-esd-growth-sweep:<project>` through `escalation_queue.submit` + `has_open_l1`. `shared/src/shared/streak.py::KeyedStreaks` (S14 λ, task 6712) is on main. The filing policy is S1's (C8 `ReconEscalation`, `policy_for`); ε consumes it |
| `fk-a88506a1e327` `harness.py` heuristic-14 | medium | leaves η1, η2 (β2 deletes the second lifecycle) | 6,418 lines; prose 3,052 (1,680 docstring + 1,372 comment-only) = 0.48 (rule: AST docstring spans + tokenize comment-only lines); 229 `task NNNN` references (rule: `[Tt]ask\s+\d{3,5}`; the same rule gives 229 at `1f240e6da2`), 27 lines naming a reviewer or an amendment (rule: `grep -c 'reviewer\|Reviewer\|amendment'`). Partition below (§Resolved design decisions D9) |
| `fk-0485729893fe` drained events lost before the run row | medium | leaf β1 (live task 6695 lands the narrow fix first) | Task 6453 closed attribution (`drain_by_ids` requires `run_id`). Still present: `BacklogIterator._run_chunk` drains, then `run_full_cycle` writes the run row; `run_full_cycle` itself drains before `journal.start_run`; a shutdown `CancelledError` in either window leaves run-attributed drained rows with no run row, which no recovery pass reads before `EventBuffer.cleanup_drained` deletes them. Task 6695 (pending) carries exactly this |
| `fk-16baf39b1db6` tests reach harness privates | medium | leaf θ (β2, γ2, η1, η2 move most of it) | 707 accesses to 55 distinct private `ReconciliationHarness` members in 16 test files (top: `_escalate` 87, `_make_stages` 60, `_run_remediation_pass` 37, `_recover_stale_runs` 35, `_fetch_filtered_task_tree` 34, `_escalation_queue` 30); 62 dotted patch targets into `fused_memory.reconciliation.harness` (`write_stage2_cycle_summary` 16, `write_stage1_cycle_summary` 13, `precheck_scope_correction_freshness` 9, `_sleep` 9). Rule: AST over `fused-memory/tests/**/*.py` mentioning "harness", `Attribute` nodes naming a single-underscore member of the class, and string constants `fused_memory.reconciliation.harness.<name>`. `stages: list[Any]` and module-level `_sleep = asyncio.sleep` still exist for tests. Tests still seed the registry by writing `harness._known_projects[…]`, although task 6158 added the read-only `known_projects` property (`36e4f5e860`). `fused-memory/tests/test_harness.py` is 22,751 lines |
| `fk-147a9951edc7` failed deferred-write replay held for the process lifetime | medium | leaf κ1 | `_replay_deferred_writes` still leaves a failed row claimed ("isn't retried in this process"), released only by `release_stale_claims(0)` at startup; exhausted rows are deleted with `logger.error`; replay passes no `causation_id` |
| `fk-10c8c43d9e89` recon self-model in drifting prompt prose | high | leaf ι1 | The premise lint and `execution_class` guard predate the pin. Still present: `prompts/stage2.py` narrates `_derive_affected_ids` / `compute_content_fingerprint` (two passages) and `prompts/__init__.py::_GUIDANCE_TOOL_PROSE['cite_task']` does too; `recon_self_model.py`'s docstring still says `MCP_CALL_SIGNATURES` and the `render_*` prose are hand-transcribed, pending a cross-check |
| `fk-20cb93d17ff3` prompts fragment library in a package `__init__` | medium | leaves ι1 (upward imports), ι2 (library) | `prompts/__init__.py` 1,392 lines; two upward imports into `server/`: `render_recon_report_tool_guidance` holds a function-local import of `fused_memory.server.recon_report`, and `prompts/stage1.py` imports `CHILD_KINDS, PARENT_ID_KEY` from `fused_memory.server.grouped_read` at module level (`CHILD_KINDS` is also declared in `memory_metadata.py`, a SPOT duplicate); stage modules import 11 / 12 / 5 names from their parent package; 21 modules and one script import from the package root (20 at `1f240e6da2`; task 9317 added `fused-memory/tests/test_recon_leak_cleanup_markup_guard.py`, which imports the package and reads `prompts.LEAKED_FRAGMENT_CLEANUP_REASON`; rule: `git grep -lE` of `from fused_memory\.reconciliation\.prompts import`, `import fused_memory\.reconciliation\.prompts` or `from fused_memory\.reconciliation import prompts`) |
| `fk-9850339b610c` stage capabilities restated as prompt literals | low | leaf ι1 | `can_file_tasks=` / `can_escalate=` literals: stage1 4, stage2 4, stage3 1; `prompts/stage2.py` names `DISALLOW_SUBTASK_CREATE`, defined nowhere in `fm/` |
| `fk-be8ed2cd9ce9` interceptor refusals classified opt-in | medium | leaf κ1 | 7 `interceptor_write_succeeded` call sites in `targeted.py` (13 mentions, counting the import and comments; rule: `grep -c 'interceptor_write_succeeded('` vs `grep -c interceptor_write_succeeded`); classification still per call site |
| `fk-05443fb78da6` interceptor attached post-construction as an Optional | medium | leaf κ1 | `TargetedReconciler.__init__` sets `self.task_interceptor = None`; `server/main.py::run_server` assigns it (and `planned_episode_registry`) after construction; `_on_task_blocked` falls back to a raw `taskmaster.update_task`; two sweeps skip when it is `None` |
| `fk-2bb6c913d747` `_on_task_done` fail-open arms invisible | medium | leaf κ1 | `_on_task_done` (450 lines) has 7 `except Exception` arms; 6 only log a warning and write no `run_actions` row (authoritative pre-check, fast-path write, flag retirement, knowledge search, planned-episode promotion, dependency check); `reconcile_task` journals `completed`; `task_interceptor.py` launches it fire-and-forget (`_on_reconciliation_done` logs exceptions only) |
| `fk-875cca636438` recon stages write tasks on the raw backend | medium | leaf κ2 | `server/main.py` constructs `ReconciliationHarness` with the raw `taskmaster`; `task_knowledge_sync.py::queue_briefing_refresh_tasks` (published by task 5914) calls `taskmaster.add_task` directly, so no `task_created` event and no curator ticket. Three raw task writers under `fm/reconciliation/` (rule: `git grep -nE '\b(taskmaster\|backend)\.(add_task\|update_task\|set_task_status\|set_task_claimant\|add_subtask)\('`): that one, the `targeted.py` `_on_task_blocked` fallback (κ1), and `consolidation_auto_executor.py`'s hazard-gate filer, which is deliberate (`plans/memory-auto-consolidation-prd.md` D8) and stays as the guard's one ratified exemption (lead ruling OQ-9) |

## Pre-conditions for activating

- Upstream stream S14 (G4): **λ 6712** `shared/src/shared/streak.py::KeyedStreaks` (done, substrate for ε);
  **μ 6721** `shared/agent_identity.py` (`AgentIdentity`, `parse_agent_identity`, `recon_stage_agent_id`;
  pending, high; it waits on S14 ν 6720). α and δ depend on 6721 (both edit `stats_verifier.py`; δ consumes
  `recon_stage_agent_id`). S14 α 6710 (`shared/project_layout.py`) is done and nothing here needs more from it.
- Substrate verified at `1b106418da` (G3): an http MCP entry handed to the Claude CLI may carry `headers`
  (`orchestrator/src/orchestrator/mcp_lifecycle.py` writes `escalation_block['headers']`, and
  `escalation/src/escalation/server.py` reads `x-escalation-identity` server-side — the same pattern δ uses);
  `server/tools.py::_resolve_identity` already reads request headers through the MCP `Context`, and every
  MCP tool that declares `agent_id` (28) calls it and returns `dict[str, Any]`, so a refusal can be the
  tool's returned error dict; on 27 of them the parameter names the caller (docstrings "Which agent is
  writing / reading / calling / …"), on `search` it is a filter ("Filter by authoring agent"); `shared/src/shared/mcp_unknown_arguments.py::unknown_argument_refusal` (task
  4867) is the house shape of a self-explaining refusal with one log line per refusal;
  `middleware/task_creation_journal.py::CreationAttribution.from_submission` is the one place task filing
  reads a submission's `_causation_id`;
  `ReconciliationHarness.__init__` receives the full `FusedMemoryConfig` (so `config.server.host/port` are in
  reach without a `server/main.py` change); `shared/src/shared/mcp_envelope.py::write_failure_reason`;
  `ReconciliationJournal.is_run_completed` (task 7230); `drained_event_restore.py::restore_unless_completed`
  (task 6453); `ReconciliationHarness.__init__`'s `escalation_queue` parameter (task 6463);
  `recon_self_model.py::FINGERPRINT_IDENTITY_FIELDS` (cross-checked against `_derive_affected_ids` by
  `fused-memory/tests/test_recon_self_model.py`); `server/recon_report.py::get_recon_report_tool_signatures`
  (a stateless function; `server/main.py::run_server` already imports `server.recon_report`);
  `shared/src/shared/mcp_post.py::mcp_endpoint_url` (the canonical slash-less MCP endpoint, which
  `orchestrator/tests/test_mcp_post_transport.py::test_no_raw_post_builds_a_trailing_slash_mcp_url` sweeps
  `fused-memory/src` for).
- Upstream sibling contracts this wave (drafts; ids wired at decompose): S1/C8
  `reconciliation/recon_escalations.py::ReconEscalation`, `policy_for` and S1/C1 `FilingPolicy` (S1 ι1; ε
  depends on it, lead ruling S12/OQ-7); S13 ι, the verbatim split of `stages/task_knowledge_sync.py` (γ0, γ1,
  ε, κ2 depend on it, lead ruling S12/OQ-8); S15 α, which moves `server/tools.py::_resolve_identity` and
  `_extract_causation` into `server/tool_context.py` (δ depends on it, lead ruling S12/OQ-11); S19 α and β1,
  which move `journal.py`, `event_buffer.py` and `recon_self_model.py` into `fused_memory/recon_state/` (α,
  κ1 depend on S19 α; δ, which now edits `recon_self_model.py`, depends on S19 β1; lead ruling S12/OQ-12).
  S10/C5 `StatusCensusUnavailable` (S10 κ2, filed) is what the census code β1 moves into the driver already
  raises.
- Live tasks that land first (§Live tasks): 7876, 6695, 7877, 8144.
- Wave-2 tasks that land first on shared files (§Cross-PRD relationship): S10 λ 9254, S10 α 9180, S26 μ
  9203, S10 μ2 9280 (and S10 μ1 9278 through it: 9280 → 9278), S10 ν 9282 (κ1, and δ, which now edits
  `middleware/task_interceptor.py`).
  Task 9285 (S10 ο, `targeted.py`) is NOT a dependency: it waits on the merge-lane chain (its 147 ancestors
  include 8528 and 7720, measured over the live dependencies table on 2026-11-12), and κ1 and 9285 edit
  different symbols of `targeted.py`, so file locks order them. Nothing in this PRD declares
  `merge_lane/worker.py`, and no task here waits on 8528, 7720 or 5040 transitively: the union of the
  ancestors of every external task dependency above (9254, 9180, 9203, 9278, 9280, 9282, 7876, 6695, 7877,
  8144, 6721) is 95 tasks and holds none of them (measured 2026-11-12). Task 4867, which that closure
  reaches through 9254, 9180, 9278, 9280 and 9282, is done (merged 2026-11-11, `daff96ef3c`), so it no
  longer holds the stream. δ's new edge onto 9282 adds four live ancestors to 9280's (54 against 50), none
  on the merge-lane chain. Against S16: S16 ε2a → S12 β1 (ε2a rebases onto `drive`); β1 never depends on
  ε2a, which waits on 8528, 7720 and 5040 through 9188.

## Sketch of approach

Three chains. In this PRD "X → Y" always means X depends on Y; a landing order is written as a list. The
**run chain** lands α, β1, β2, γ1, γ2, δ, ι1, ε, ζ, κ1, κ2, η1, η2, θ in that order, each depending
on the one before, because each edits `harness.py`: it types the run record, builds the one driver, moves
remediation onto it, changes the stage interface, binds stage identity on the transport, reduces the streaks
to declarations, types the cycle-summary state, routes recon task writes through the interceptor, then
partitions `harness.py` and retires the private test seams. γ0, which moves the stage tests onto one builder
and edits test files only, stands off the chain: it waits only on S13 ι, and γ1 depends on it because γ1
changes the helper module γ0 creates (tasks that share only test files are not chained; the lead's
test-file rule). ω, the integration gate, depends on θ and ι2. The **prompt pair** (ι1 → δ, which brings γ2's collaborators; ι2 → ι1)
derives stage capabilities and the dedup rule from code and moves the prompt fragments out of the package
`__init__`. The **task-write pair** (κ2 → κ1) gives the targeted reconciler and the stages one gated write
path. ω is the integration gate.

**α — typed run record.** `ReconciliationRun.stage_reports: dict[StageId, StageReport]` and
`error: RunError | None` (C4). `RunError` is a frozen model with one constructor from an exception (closing
task 5912's ask) and the deferred flag. `journal.py::_row_to_run` is the only place a legacy row becomes this
type: a stored `_error` becomes `error`; `_resume` and any other `_`-prefixed key are carried opaquely in
`legacy_meta` and written back verbatim (so `citation_repair.py`'s decode → mutate →
`compare_and_set_run_stage_reports` round trip never erases them); a stage-keyed dict without `'stage'` (the
remediation short-circuit's synthetic `{'items_flagged': [...]}`) becomes a `StageReport` with
`agent_outcome=not_launched` and the run's `started_at`/`completed_at` as its timestamps. Serialization keeps
the persisted JSON shape (`_error` key included) so existing rows, `get_run_stage_execution` and
`MemoryService.get_cycle_summary_presence` read unchanged. Readers drop their shape branches.

**β1/β2 — the one driver.** `ReconciliationHarness.drive(request: RunRequest) -> ReconciliationRun` (C1) owns
the whole lifecycle, inside `_active_runs.track(run_id, …)` for every kind, in this order: the kind's
pre-step; `journal.start_run`; then, inside the ladder's `try`, taking events (drain, drain by ids, or none)
and recording `events_processed`; diagnostics and stage inputs; the stage loop with the pre-Stage-3 flush; the
watermark (kinds that advance it); `complete_run('completed')`; the post-completion tail (verify, persist, run
streaks, judge, kind tail), which catches its own failures; then the one except ladder for failures before
durable completion; then the `finally` (flush, persist, GC). This is today's `run_full_cycle` order
(`_finish_durably_completed_cycle` runs inside the track and before the `finally`), so the `recon_busy`
window the cycle-aware restart reads is unchanged. `run_full_cycle` and `_run_remediation_pass` become
one-call wrappers that build a `RunRequest`; `BacklogIterator._run_chunk` builds a `backlog_chunk` request
with the chunk's event ids and calls `drive`, and stops draining and restoring itself. β1 lands the driver for
`full` and `backlog_chunk`; β2 moves `remediation` onto it and deletes the second lifecycle (~800 lines).

**γ0 — one stage test builder.** Before the stage interface changes, the 34 test modules that construct or
run a stage move onto `fused-memory/tests/reconciliation/stage_builders.py::build_stage` / `run_stage`, a
uniform substitution with no production change, so γ1 and γ2 change one helper instead of 34 modules.

**γ1/γ2 — stage interface.** `BaseStage.run(events, watermark, prior_reports, run: RunContext)` and
`assemble_payload(events, watermark, prior_reports, run: RunContext) -> PayloadResult` (C2). `BaseStage.run`
merges `PayloadResult.stats` into the report and hands `PayloadResult.carry` to the stage's
`after_report(report, payload, run)` hook, which replaces every `run()` override's read-back of a stash. γ2
constructs each stage with its collaborators and its per-cycle inputs (a frozen `Stage1Inputs` /
`Stage2Inputs` / `Stage3Inputs`) and the escalation endpoint; `_configure_consolidator`,
`_configure_task_sync`, `_propagate_escalation_queue` and the 17 harness-set attributes go.

**δ — identity on the transport, mismatches refused, the instructions swept.** The stage's fused-memory MCP
entry is built from the run context: its URL is the server's own, `mcp_endpoint_url(f'http://127.0.0.1:{port}')`
from `config.server` (or `host` when it is not `0.0.0.0`), and it carries two headers naming the stage
identity and the run id (C3). The stage's recon-report entry carries the same two headers, for S13's
recon-report leaf to read under the same rule. On the server, the headers supply the identity and the
causation id; a tool call whose caller `agent_id` argument (every tool but `search`, whose `agent_id` is a filter) or
`metadata['_causation_id']` differs from its header is refused before the tool touches a store, with a structured error that names both values and says the
connection already identifies the stage and run, and one log line per refusal. Task filing reads the run id
the same way, so Stage 2's creation-journal rows carry the run without the model typing it. In the same
commit every instruction a stage reads to send those values goes: the prompt's "include agent_id /
_causation_id" paragraph becomes the explicit line "The connection already identifies you and this run. Do
not pass `agent_id` or `_causation_id`."; the Stage-1 and Stage-2 examples and the self-model's accounting
text stop asking for them; `CLAUDE.md`, the `/memory` command, `docs/task-authoring.md` and the server's MCP
instructions string gain the one-line exception for reconciliation stages. `_find_fused_memory_server` goes.

**ε — streaks.** `reconciliation/run_streaks.py` declares each "N consecutive completed full cycles whose
stage stat reads bad" alarm as a `RunStreak` value and evaluates all of them in the driver's post-completion
tail by replaying the journal window through `KeyedStreaks` (C5). Each alarm is a member of S1's recon
escalation registry (S1/C8 `ReconEscalation`) and files through `policy_for(member)`; this PRD chooses the
policy only. Both quartets and the stage-side filing go.

**ζ — cycle-summary write state.** `StageReport.cycle_summary: CycleSummaryWrite | None` replaces the three
stats flags; the stage writers and the harness backstop arms set it; the predicate reads it.

**ι1/ι2 — prompts.** `reconciliation/stage_capabilities.py` derives each stage's capabilities from the denial
tables, which stay in `cli_stage_runner.py` (`stage_capabilities.py` imports them); prompts render capability
text from them; the dedup-anchor rule is rendered from `recon_self_model.FINGERPRINT_IDENTITY_FIELDS`; a drift
test cross-checks `MCP_CALL_SIGNATURES` against the live tool registries. The recon-report tool guidance string
is computed by `server/main.py::run_server` (which already imports `server.recon_report`) and passed to the
`ReconciliationHarness` constructor, which hands it to stages through `StageCollaborators.report_tool_guidance`;
`prompts/stage1.py` takes `CHILD_KINDS` and `PARENT_ID_KEY` from `fused_memory/memory_metadata.py` (which
`server/grouped_read.py` then re-binds from). So neither `prompts/` nor `harness.py` imports anything from
`server/` (ι1). ι2 moves the fragment library to `prompts/fragments.py` and repoints its importers, with no
re-export left behind.

**κ1/κ2 — task writes.** `TaskInterceptor` builds its `TargetedReconciler` through a provider member of S10
μ2's frozen `InterceptorCollaborators` bundle, resolved with the interceptor itself as the reconciler's write
gate at the end of construction, so the reconciler can never exist without one (C6). One
`_gated_write` classifies every reconciler write through `write_failure_reason`; each fail-open arm of
`_on_task_done` writes a `run_actions` row, counted in aggregate by `ReconciliationJournal.get_stats`; a failed
deferred-write replay releases its claim with a bounded attempt count and carries its causation id. κ2 hands
the harness and its stages the interceptor as their task-write port, so the briefing-refresh writer submits
through the curator, and a guard test fails on any raw-backend task write under `fm/reconciliation/` except the
one ratified exemption, `consolidation_auto_executor.py`'s hazard-gate filer.

**η1/η2/θ — partition and test seams.** After the behaviour leaves, `harness.py` splits along its measured
partition (D9) into modules that import downward only; the tests of each moved part move to a test module of
their own and drive its public API; θ retires the remaining private seams.

## Resolved design decisions

- **D1 — RunKind is a value on one driver (R12).** `RunKind.full`, `RunKind.backlog_chunk` and
  `RunKind.remediation` select a row of one policy table (C1): which event source, whether diagnostics run,
  whether the watermark advances, the pre-step, the failure category, and the kind tail. `RunKind` maps to the
  persisted `RunType` (`full` and `backlog_chunk` persist as `RunType.full`, as backlog chunks do today, so the
  streak window and every journal reader are unchanged). A new run kind is one table row.
- **D2 — the one cancellation rule (the R12 decision).** Task 5514 already removed the resume half of the
  asymmetry. The rule for every kind: a run cancelled before durable completion is journaled `failed` with
  `RunError(error_type='CancelledError', failed_stage=…)`, its own drained events are restored unless the
  journal records it completed, and the cancellation propagates. A run the journal records completed is never
  relabelled — the guard task 7230 added for `run_full_cycle` (`_end_run_unless_completed`) becomes the only way
  any arm ends a run. Rejected: letting remediation swallow cancellation, which would keep a shutdown waiting on
  a cancelled pass; it is cancellation that ends the parent's tail, and the parent's tail already propagates it.
- **D3 — one except ladder.** `CancelledError`: D2, re-raise. `AllAccountsCappedException`: `failed`,
  `RunError(deferred=True)`, restore, return the run. `Exception`: `failed`, restore, file the kind's failure
  category (`recon_failure` for full and backlog chunk, `recon_integrity_issue` for remediation — today's
  categories, now a table column), re-raise. Remediation re-raising files once: its caller is
  `harness.py::ReconciliationHarness._maybe_remediate`, whose `try … except Exception` today files a second
  `recon_integrity_issue` (`'Remediation orchestration failed: …'`) for the PARENT run id, a different
  fingerprint that would not fold. β2 narrows that `try` to `_maybe_remediate`'s own orchestration
  (partitioning, suppression, the deferral gate) and puts the `drive(RunRequest.remediation(…))` call after it,
  so the driver's ladder is the one filer of a remediation failure; the raise then reaches the parent's
  post-completion tail, whose existing `except Exception` logs `reconciliation.post_completion_tail_failed` and
  never relabels or files for the parent (boundary row 17).
- **D4 — the run row exists before any event is drained.** The driver writes `journal.start_run` and only then
  drains (`buffer.drain(run_id=…)` for `full`, `buffer.drain_by_ids(ids, run_id=…)` for `backlog_chunk`).
  A cancellation anywhere after that leaves a `running` row, which `_recover_stale_runs` (through `_recover_one_run`)
  already restores run-scoped. `BacklogIterator` records the chunk boundary before calling `drive` and no
  longer drains or restores. Task 6695 lands the narrow fix first; D4 keeps it. Because
  `journal.py::ReconciliationJournal.start_run` is today the only writer of `runs.events_processed` (read by
  `dashboard/src/dashboard/data/reconciliation.py`, `reconciliation/throughput.py` and the judge prompt), the
  driver records the count for `full` through a new `ReconciliationJournal.set_events_processed(run_id, n)`
  right after the drain, inside the same shielded section; `backlog_chunk` passes `len(ids)` to `start_run`.
- **D5 — scope freshness is the remediation pre-step, and a short-circuit is a real report.** The pre-step
  runs after `start_run`; when every finding is confirmed fresh it ends the run `completed` with
  `stage_reports[StageId.integrity_check] = StageReport(items_flagged=skipped, agent_outcome=not_launched)`, so
  `_finding_persistence_count` counts it exactly as today without a dict branch.
- **D6 — RunContext carries StageId; PayloadResult refines R12's (text, stats).** `RunContext` (C2) is frozen
  and built by the driver once per stage invocation. `PayloadResult(text, stats, carry)`: `stats` is merged into
  the report by `BaseStage.run`; `carry` is a stage-typed frozen value for the two outputs that are not stats
  (`MemoryConsolidator`'s live-workflow snapshot, which the citation guard reads; `TaskKnowledgeSync`'s
  combined flags). Without `carry` those two would stay stashes, which R12 forbids. Stage-specific `RunContext`
  fields are not added; per-stage inputs belong to γ2's inputs values.
- **D7 — stage identity and run id are bound on the transport; a differing value is refused (R12 "StageId in
  the run context"; sitting ruling OQ-3).** The harness, not the LLM, names the caller and the run: two
  headers on the stage's fused-memory MCP entry and on its recon-report entry (C3). A matching or absent
  argument lands, attributed to the header. An `agent_id` or `_causation_id` that differs from its header is
  refused, not overridden: once the prompt says not to send either, a mismatch can only come from a model
  ignoring that line or copying an old run id, and Leo's fail-loud rule says that should be seen, not
  corrected without a trace. The refusal is one function at one place in the server (C3), applied before any
  store is touched. The ruling names writes; the three read tools whose `agent_id` names the reader
  (`get_entity`, `get_entity_by_uuid`, `get_episodes`: "Which agent is reading") follow the same rule,
  because the resolver is one function and a mismatch there has the same cause (confirmed by the lead; a
  writes-only reading would give the resolver a read/write switch). `search` is outside the rule: its
  `agent_id` is a filter ("Filter by authoring agent"; the `/memory` command's "By agent" pattern,
  `agent_id="claude-task-7"`, names another writer), so a differing value there is not a mismatch; the
  header neither refuses nor replaces it and `search` keeps today's resolution for that argument (confirmed
  by the lead; S15's verbatim tool-family moves carry the exclusion through). Measured over the 7 days to 2026-11-12 (`digests/discussion/S12.md` §S12-2): 1,255
  run-tied writes carried a `recon-stage-*` id, 5,070 carried none, none carried another id, so the refusal
  would have blocked nothing on that traffic. Rejected: the header winning with a
  `recon_stage_identity_override` log event (the authored option; it corrects a stale echoed run id
  silently), and keeping the prose and validating the echoed value (it still depends on the LLM echoing
  something). Every other client sends no header and keeps passing `agent_id` exactly as today; the
  parameter stays in all tool signatures.
- **D7a — the instruction sweep lands with the refusal (sitting ruling OQ-3, Leo's addition).** Every stage
  CLI session runs with its working directory in the dark-factory checkout and loads `CLAUDE.md` and the
  `/memory` command (S12.md §S12-2, from 200 of 1,338 archived stage transcripts; the stage launcher passes no
  setting-source restriction). Both say to always pass `agent_id`, with example values a stage would be
  refused for. So the sweep cannot follow the refusal: δ makes both changes in one commit. The stage prompt
  states the rule positively ("The connection already identifies you and this run. Do not pass `agent_id`
  or `_causation_id`."), because silence would leave the "always pass" rule standing. The sweep's file set
  is closed: the six files S12.md §S12-2 lists, plus `docs/task-authoring.md` §"Write-tagging convention",
  which carries the same rule and the same `"reconciliation-stage-1"` example and which `CLAUDE.md` names as
  the task-authoring reference (the fold found it; confirmed by the lead), plus
  `server/tools.py::FUSED_MEMORY_INSTRUCTIONS`, the fused-memory server's MCP instructions, whose
  "Include agent_id for attribution (e.g. \"claude-interactive\", \"claude-task-7\")" convention every stage
  session receives with the tool list (the fold-verify grep at `0f21f5e457` found it; same file as the
  refusal, no new file; confirmed by the lead). That grep (`agent_id|_causation_id|recon-stage-` over the stage
  prompt sources, `agent_id` over `CLAUDE.md`, `.claude/`, `docs/`, `skills/`, the server's instructions and
  tool docstrings, `~/.claude/CLAUDE.md` and the project auto-memory index) found no other site that tells a
  stage to send an id: the tool docstrings of `add_system_record` and `propose_consolidation` name the
  required `recon-stage-` value but already mark the argument "optional, auto-derived from MCP context", and
  a stage that omits it satisfies them through the header. Unchanged by design: prompt lines that
  describe reading a stored record's `agent_id` or what the server rejects, the `add_memory` signature in
  `recon_self_model.py::MCP_CALL_SIGNATURES` (the parameter still exists), and every skill, project skill
  overlay and orchestrator agent prompt (their readers send no header).
- **D8 — streaks are declarations replayed through `KeyedStreaks` (R12 "over S14's shared counter").** The
  journal stays the durable source (a restart does not reset a streak); the replay feeds a fresh
  `KeyedStreaks(threshold)` the leading run of bad cycles oldest-first plus the current one, and files whenever
  the current step's `count >= threshold` (`StreakStep.count`), letting the filer's open and recently-resolved
  fold absorb the repeats, so a resolved-but-persisting condition files again on the first bad cycle after the
  24-hour recently-resolved window (`harness.py::_RESOLVED_RECURRENCE_WINDOW_SECONDS`), instead of never until a
  good cycle (sitting ruling OQ-1). A `None` reading stops the run
  (today's rule in both copies). Each alarm is one `ReconEscalation` member of S1's registry (S1/C8, with
  `folds=True`) and files through `policy_for(member)` (S1/C1 `FilingPolicy`); this PRD chooses the member and
  adds no level, severity or dedupe parameter to `_escalate` (G4: recon chooses a policy only). ε moves the
  growth alarm's evaluation from inside Stage 2 to the driver's post-completion tail: a cycle whose Stage 3
  fails, which never reaches durable completion, no longer evaluates it (its stat is journaled, but a failed
  run is outside every later window).
- **D9 — the `harness.py` partition (heuristic 14 protocol).** Prose first: reviewer-answering prose (the
  "WHY THE GUARD", "WHY BEFORE STAGE 3 / WHY AT THE TOP / WHY THIS IS SAFE", "Reviewer finding …", "Known
  trade-off …" blocks and their kin) moves to the commit messages of the leaves that touch it, leaving at most a
  one-line pointer (Comments stance). Partition, measured at `1f240e6da2` (lines of the named symbols; the
  only change to `harness.py` since, task 6158, touches the imports and the constructor and adds the
  read-only `known_projects` property, all in the wiring that stays, so every moved span holds at
  `1b106418da`):
  run driver 519 (stays — R12: `harness.py` owns the driver); remediation 1,181 (β2 deletes ~800, the
  remediation tail stays with the driver as its kind tail); **cycle-summary backstop 766, moved to
  `reconciliation/cycle_summary_backstop.py`** (η1; imports `journal`, `summary_pool`, `models` downward, never
  `harness`); **run recovery 363, moved to `reconciliation/run_recovery.py`** (η1; `journal`, `event_buffer`,
  `drained_event_restore`, `cli_stage_runner.gc_run_config_dir`); **cycle diagnostics 506, moved to
  `reconciliation/cycle_diagnostics.py`** (η2; census, status correction, queue and index health; needs
  `memory_service`, the task port and `index_health`); **finding gate and filing 765, moved to
  `reconciliation/finding_gate.py`** (η2; the persistence count, actionability helpers and
  `_file_finding_task_escalation`, beside the existing `finding_task_escalation.py`); **`BacklogIterator` 192, moved to
  `reconciliation/backlog_iterator.py`** (η2; it calls `harness.drive` and the public `known_project_scope_for`,
  and files through the injected recon filer — the reach-ins to `_escalate` and `_known_project_scope_for` go).
  Heuristic-13 check per candidate: none imports `harness` (each takes collaborators at construction and the
  harness constructs it), so none closes a cycle or reaches back. The loop, judge, escalation-server and stage
  wiring stay (≈ 1,100 lines). Estimated remainder ≈ 2,400 lines before prose rehoming (not a target; the done
  criterion is the partition, and η2 records the stays-whole measurement for the remainder in its commit).
- **D10 — one recon task-write rule, one ratified exemption.** Every task write that reconciliation makes goes
  through the TaskInterceptor (C6); the raw backend is a read port only, with one deliberate exemption: no
  reconciliation module except `reconciliation/consolidation_auto_executor.py` writes a task through the raw
  backend. That module's hazard-gate filer stays on the raw backend because
  `plans/memory-auto-consolidation-prd.md` D8 decided it (no curator LLM call, no candidate-key dedup, no
  premise or markup guard), and this PRD does not reverse that decision (lead ruling OQ-9). A guard test fails on a call
  to `add_task`, `update_task`, `set_task_status`, `set_task_claimant` or `add_subtask` on anything but the
  write port under `fm/reconciliation/`; its exemption list holds that one module and is an INV-12 governed
  list whose entry is `Policy` row `inv12-recon-hazard-gate-raw-backend` in
  `docs/legibility/exception-ratifications.yaml` (source: `plans/memory-auto-consolidation-prd.md` D8 and the
  lead's ruling S12/OQ-9 of 2026-11-11, flagged to Leo). The briefing-refresh writer submits through
  `submit_task` and records ticket ids (lead ruling OQ-4).
- **D11 — tests follow the partition.** Each extracted module gets its own test module that drives its public
  API; the moved tests leave `fused-memory/tests/test_harness.py` (22,751 lines). A leaf that retires a private
  seam names the public seam the tests move onto (tests stance).
- **D12 — per-file order against wave 2.** On every production file a wave-2 task also edits, this stream's
  first task on that file depends on the last wave-2 task on it (§Cross-PRD relationship; lead ruling OQ-6), except
  `targeted.py`, where the last one (S10 ο 9285) waits on the merge-lane chain and edits different symbols, so
  file locks order it (§Pre-conditions). An edge that rests only on a shared test file is not wired (the
  lead's test-file rule, inside this plan and across plans): test files are rebase-only, the scheduler's file
  locks serialize them and the later task rebases. So S3 δ 9167, S3 ε1 9188 and S3 ε2 9193 on
  `fused-memory/tests/test_stages.py` (their `_REVIEWED_STAGE_SAFE` edits) and S10 μ1 9278 and μ2 9280 on
  `fused-memory/tests/test_auto_consolidation_e2e.py` (which γ0 also edits) take no edge.
- **D13 — one decision partition per decision area (Leo, 2026-11-12, cross-stream; invariant
  `one-decision-partition-per-area`, INV-23).** Each table or rule
  this plan builds, with who decides each row:

  | Decision area | Decided by code (one place) | LLM judgement, and its path to a human |
  |---|---|---|
  | How a run of each kind starts, takes events and ends (D1–D4) | the `RunKind` policy table and the one except ladder in `drive` (C1) | none |
  | Which stage and run a write belongs to (D7) | C3's resolver over the two headers; a differing argument is refused by code. This row moves a decidable fact OUT of the LLM: today the prompt asks the model to echo it | none |
  | What a stage may do (ι1) | `capabilities_for(stage)` over the denial tables (C7) | none |
  | When a streak alarm files (D8) | `RunStreak` declarations, `KeyedStreaks` and S1's `policy_for` (C5) | the recon escalation watcher judges the filed record (cause, fix or promote), and promotes to a human |
  | Whether a targeted-reconciler write landed; what a fail-open arm leaves (κ1) | `write_failure_reason` in `_gated_write`; one `run_actions` row per arm (C6) | none |
  | Whether a recon task write may use the raw backend (D10) | the guard test and its one ratified exemption | none |
  | A failed remediation pass (D3) | the ladder files one `recon_integrity_issue` | the recon escalation watcher, then a human |

  No row hands a decidable case to an LLM. The stage agents' own work (what to consolidate, which task to
  file) is the judgement reconciliation exists for and is not a table this plan builds. Evals are hermetic
  (cross-stream rule): not engaged. No eval cell constructs `ReconciliationHarness` (only
  `server/main.py::run_server` does, outside tests), and this plan adds or moves no store, sink, filer,
  queue or archive outside fused-memory's reconciliation path; κ1's `run_actions` rows and ω's read-only
  script stay in `data/reconciliation/reconciliation.db`.

## Contracts

```python
# C1 — the one run driver  (fm/reconciliation/run_context.py types; driver in harness.py)   β1, β2
class TierConfig: ...                           # moved by β1 from harness.py into run_context.py; harness.py imports it
                                                # (fields unchanged; RunRequest.tier carries whatever _select_tier returns,
                                                #  including after S16's routing leaf rewrites _select_tier)

@dataclass(frozen=True)
class TreeSnapshot:                             # β1
    tree: FilteredTaskTree
    read_at: datetime                           # the instant the harness read the tree (tasks 4115, 5093, 6027)

class RunKind(StrEnum):
    FULL = 'full'; BACKLOG_CHUNK = 'backlog_chunk'; REMEDIATION = 'remediation'
    @property
    def run_type(self) -> RunType: ...          # FULL, BACKLOG_CHUNK -> RunType.full; REMEDIATION -> RunType.remediation

@dataclass(frozen=True)
class EventSource:                              # exactly one of three shapes
    @classmethod
    def drain(cls) -> 'EventSource': ...        # full: buffer.drain(project_id, run_id=...)
    @classmethod
    def by_ids(cls, ids: Sequence[str]) -> 'EventSource': ...   # backlog chunk: buffer.drain_by_ids(..., run_id=...)
    @classmethod
    def none(cls) -> 'EventSource': ...         # remediation

@dataclass(frozen=True)
class RunRequest:
    kind: RunKind
    project_id: str
    trigger_reason: str
    tier: TierConfig
    events: EventSource
    run_id: str                                 # caller-minted (task 6453 keeps it)
    assembled_payload: AssembledPayload | None = None     # backlog chunk only
    parent_run_id: str | None = None            # remediation only
    findings: tuple[dict, ...] = ()             # remediation only
    tree: TreeSnapshot | None = None            # (tree, read instant) the parent already fetched
    @classmethod
    def full(cls, ...) -> 'RunRequest': ...
    @classmethod
    def backlog_chunk(cls, ...) -> 'RunRequest': ...
    @classmethod
    def remediation(cls, ...) -> 'RunRequest': ...

class ReconciliationHarness:
    async def drive(self, request: RunRequest) -> ReconciliationRun: ...
    async def run_full_cycle(self, project_id, trigger_reason, tier=None, events=None,
                             assembled_payload=None, run_id=None) -> ReconciliationRun: ...   # builds a request; calls drive
    def known_project_scope_for(self, project_id: str) -> ProjectScope: ...                  # public; BacklogIterator uses it

class ReconciliationJournal:                    # β1 (journal.py, at its recon_state path; lead ruling OQ-12)
    async def set_events_processed(self, run_id: str, n: int) -> None: ...   # the only writer besides start_run
```

Invariants (C1):
0. `drive` wraps every kind's lifecycle in `_active_runs.track(run_id, …)`, so a remediation pass is tracked as
   its own run; the order is pre-step, `start_run`, `try:` [drain → `events_processed` → stages →
   `complete_run('completed')` → post-completion tail], the except ladder, `finally:` [flush, persist, GC] —
   the post-completion tail runs before the `finally`, as `_finish_durably_completed_cycle` does today.
1. `journal.start_run(run)` precedes every drain; the drain is the first statement inside the ladder's `try`,
   so a cancel that lands during the drain meets invariants 2 and 3; a kind with `EventSource.none()` drains
   nothing.
2. Every arm ends a run through one guard: nothing is written for a run the journal records completed, and
   the in-memory run is put back to `completed` (D2).
3. Cancellation: `failed` + `RunError(CancelledError)` + own events restored unless completed; re-raised.
4. `AllAccountsCappedException`: `failed` + `RunError(deferred=True)` + restore; the run is returned.
5. `Exception`: `failed` + restore + the kind's failure category filed (filed before the bookkeeping writes,
   as task 5545's fix requires); re-raised.
6. `finally`: `_flush_cycle_summaries` then persist-and-GC, for every kind.
7. The post-completion tail runs only after durable completion; its failure is logged as
   `reconciliation.post_completion_tail_failed` and never relabels, restores or files `recon_failure`.
8. Only `full` and `backlog_chunk` advance the watermark and acknowledge overreach flags; only `full` and
   `backlog_chunk` evaluate run streaks; only `full` and `backlog_chunk` call `_maybe_remediate`;
   remediation's kind tail is the persistence-gated escalation and never a third pass.
9. `BacklogIterator` neither drains nor restores; on its deadline the driver's cancellation arm restored the
   in-flight run's events before `TimeoutError` reaches the iterator.
10. A `full` run's row carries the drained count (`ReconciliationJournal.set_events_processed(run_id, n)`,
    written inside the shielded section right after the drain) before the first stage starts; a
    `backlog_chunk` run's row carries `len(ids)` from `start_run`; a `remediation` run's row carries 0.
11. A remediation failure is filed exactly once, by the driver's ladder, keyed on the remediation run id:
    `_maybe_remediate` calls `drive` outside its own `try` (D3).

Errors: `UnknownProjectError` from `known_project_scope_for` before any side effect (task 1143, unchanged).

```python
# C2 — RunContext and the stage interface  (fm/reconciliation/run_context.py)   β1 (type), γ1 (stages), γ2 (inputs)
@dataclass(frozen=True)
class RunContext:
    run_id: str
    kind: RunKind
    scope: ProjectScope
    stage: StageId                              # the stage this context is handed to
    model: str | None
    window_start: datetime | None               # the run window a stage reads (today TaskKnowledgeSync._run_window_start)
    cycle_fence: datetime                       # cycle start; today MemoryConsolidator.cycle_fence_time
    tree_read_at: datetime | None               # instant the harness read the task tree (tasks 4115, 5093, 6027)
    def for_stage(self, stage: StageId) -> 'RunContext': ...

@dataclass(frozen=True)
class PayloadResult(Generic[C]):
    text: str
    stats: Mapping[str, int | float | bool | str | list]   # merged into StageReport.stats by BaseStage.run
    carry: C | None = None                                  # stage-typed non-stat output (D6)

class BaseStage:
    def __init__(self, stage_id, collaborators: StageCollaborators, inputs: StageInputs, *, scope): ...  # γ2
    async def run(self, events, watermark, prior_reports, run: RunContext) -> StageReport: ...           # γ1
    async def assemble_payload(self, events, watermark, prior_reports, run: RunContext) -> PayloadResult: ...  # γ1
    async def after_report(self, report: StageReport, payload: PayloadResult, run: RunContext) -> None: ...   # γ1; default no-op

@dataclass(frozen=True)
class ReconReportEndpoint:                      # γ2
    port: int
    state: object                               # the ReconReportState the harness holds today
@dataclass(frozen=True)
class EscalationEndpoint:                       # γ2; replaces BaseStage._escalation_url / _escalation_queue
    url: str
    queue: EscalationQueue

@dataclass(frozen=True)
class StageCollaborators:                       # γ2; the same for every stage of a run
    memory: MemoryService; tasks: TaskBackendProtocol | None; task_writes: ReconTaskWritePort | None  # task_writes: κ2
    journal: ReconciliationJournal; config: ReconciliationConfig; usage_gate: UsageGate | None
    known_projects: Mapping[str, str]; recon_report: ReconReportEndpoint; escalation: EscalationEndpoint | None
    fused_memory_mcp_url: str                   # δ
    report_tool_guidance: str                   # ι1; rendered by server/main.py::run_server, passed to the harness constructor

@dataclass(frozen=True)
class Stage1Inputs: tier: TierConfig; prior_s3_findings: tuple[dict, ...]; remediation_findings: tuple[dict, ...] | None
    # assembled_payload, tree, task_count_verification, graphiti_queue_health, index_health,
    # status_correction_reconciliation, overreach_flags — the 13 attributes of today, as fields
@dataclass(frozen=True)
class Stage2Inputs: tree: TreeSnapshot | None; remediation_mode: bool
@dataclass(frozen=True)
class Stage3Inputs: tree: TreeSnapshot | None
StageInputs = Stage1Inputs | Stage2Inputs | Stage3Inputs   # γ2
```

Every type C1 and C2 name is defined here or already exists (`FilteredTaskTree`, `ProjectScope`,
`AssembledPayload`, `EscalationQueue`, `MemoryService`, `ReconciliationJournal`, `ReconciliationConfig`,
`UsageGate`, `TaskBackendProtocol`); `run_context.py` imports none of them from `harness.py`, which imports
`run_context.py`. C1 and C2 share one module (lead ruling OQ-5).

Invariants (C2): a stage instance serves exactly one run (the harness builds stages per run, as today); no
stage attribute is assigned after construction; no stage method reads a value another method wrote on `self`
during the same run; `assemble_payload` raises nothing about a missing run id (the parameter is required).
Exact field lists beyond those named are tactical.

```python
# C3 — stage identity on the transport; a differing value is refused  (fm/reconciliation/run_context.py;
#      fm/server/tool_context.py reads and refuses)   δ
STAGE_AGENT_ID_HEADER = 'X-Fused-Memory-Agent-Id'
CAUSATION_ID_HEADER = 'X-Fused-Memory-Causation-Id'
def stage_mcp_headers(run: RunContext) -> dict[str, str]:
    # {STAGE_AGENT_ID_HEADER: shared.agent_identity.recon_stage_agent_id(run.stage), CAUSATION_ID_HEADER: run.run_id}

# the entry URL: shared.mcp_post.mcp_endpoint_url(f'http://127.0.0.1:{config.server.port}')
#   (config.server.host instead of 127.0.0.1 when it is not '0.0.0.0'); no new ServerConfig property

# fm/server/tool_context.py: S15's ToolContext leaf (S15 α) moves the two helpers there before δ; under the lead's
#   ruling OQ-11 (a1) δ also passes ctx at the 17 _extract_causation call sites in fm/server/tools.py, and S15's
#   gate-protocol leaf (S15 δ → S12 δ) takes the causation id from this resolver, so the refusal survives it
@dataclass(frozen=True)
class StageIdentityRefusal:                    # the one structured refusal; what the caller reads
    field: Literal['agent_id', '_causation_id']
    header: str                                 # STAGE_AGENT_ID_HEADER or CAUSATION_ID_HEADER
    header_value: str
    argument_value: str
    def as_error(self) -> dict[str, str]: ...   # {'error_type': 'stage_identity_mismatch', 'error': <message>, 'field', 'header',
                                                #  'header_value', 'argument_value'}; the message names both values and says
                                                #  "the connection already identifies this stage and run; omit <field>"
def stage_identity_refusal(ctx, *, agent_id: str | None, causation_id: str | None) -> StageIdentityRefusal | None: ...
    # the ONE predicate: a header that is present and a non-blank argument that differs from it -> a refusal
    # (agent_id checked first); called by the two resolvers below and by S13 ξ's recon-report resolver
    # server/recon_report.py::_report_run_id (which widens `field` by 'run_id' and re-labels the refusal with
    # dataclasses.replace; one predicate, one error shape); never for search's agent_id filter
def _resolve_identity(agent_id, session_id, ctx) -> ...      # header present: argument absent or equal -> the header's value;
                                                             #   differing -> refused (StageIdentityRefusal); header absent -> today's rule
def _extract_causation(metadata, agent_id, ctx) -> ...       # the same rule for CAUSATION_ID_HEADER against metadata['_causation_id']

# fm/middleware/task_creation_journal.py
class CreationAttribution:
    @classmethod
    def from_submission(cls, metadata, agent_id, *, bound_causation_id: str | None = None) -> 'CreationAttribution': ...
        # bound_causation_id (the run-id header, passed down by submit_task through the interceptor) is the attribution
        # when given; metadata['_causation_id'] otherwise (every non-recon submitter, unchanged)
```

Invariants (C3): the stage's fused-memory MCP entry is `{'type': 'http', 'url': mcp_endpoint_url(…), 'headers':
stage_mcp_headers(run)}`; the stage's `'recon-report'` entry in `stages/base.py::BaseStage._build_mcp_config`
carries the same `stage_mcp_headers(run)` (lead ruling OQ-10); with a header present, a matching or absent
argument lands attributed to the header, and a differing `agent_id` or `_causation_id` is refused before the
tool reads or writes any store, the caller receiving `StageIdentityRefusal.as_error()` and the server writing
one `recon_stage_identity_refused` log line (both values, the tool, the field) per refusal; the refusal is
decided only by `stage_identity_refusal`, so the same rule holds for `agent_id` on every tool where it names
the caller (27 of the 28 that declare it, three reads among them; `search`, whose `agent_id` filters by
authoring agent, keeps today's resolution for that argument and is never refused on it) and for `_causation_id` wherever the
server reads one (the 17 `_extract_causation` call sites and, after δ, `submit_task`), measured at
`1b106418da`;
`submit_task` resolves the run id through `_extract_causation`'s rule and passes it to the interceptor as
`bound_causation_id`, so a stage's creation-journal row carries its run although the prompt no longer asks
for `_causation_id`; absent headers leave today's behaviour exactly (every non-recon client: task agents,
interactive sessions, scripts, skills); no `recon_stage_identity_override` event exists; the stage prompt
carries the line "The connection already identifies you and this run. Do not pass `agent_id` or
`_causation_id`." and no instruction to send either; the prompt keeps `run_id` for recon-report calls until
S13's recon-report leaf reads `CAUSATION_ID_HEADER` under the same refusal rule and drops it.

```python
# C4 — the typed run record  (fm/models/reconciliation.py; journal.py rehydrates)   α, ζ
class RunError(BaseModel, frozen=True):
    error_type: str; error_message: str; failed_stage: StageId | None
    traceback: str = ''; deferred: bool = False; sqlite: Mapping[str, object] = {}
    @classmethod
    def from_exception(cls, exc: BaseException, *, failed_stage, message: str | None = None, deferred=False) -> 'RunError': ...

class ReconciliationRun(BaseModel):
    stage_reports: dict[StageId, StageReport]
    error: RunError | None = None
    legacy_meta: Mapping[str, object] = {}       # every legacy `_`-prefixed key except `_error`, carried opaquely
    # other fields unchanged

class CycleSummaryWrite(StrEnum):                # ζ
    WRITTEN = 'written'; WRITE_FAILED = 'write_failed'
    RECOVERED = 'recovered'; RECOVERY_UNCONFIRMED = 'recovery_unconfirmed'; DEGRADED_SYNTH = 'degraded_synth'
class StageReport(BaseModel):
    cycle_summary: CycleSummaryWrite | None = None   # None = the stage never reached its write
```

Invariants (C4): `journal.py::_row_to_run` is the only legacy-row decoder (a `_error` blob becomes `error`; any
other `_`-prefixed key, `_resume` included, goes into `legacy_meta` verbatim; a stage dict without `stage`
becomes `StageReport(stage=key, agent_outcome=not_launched)` with `started_at`/`completed_at` taken from the
run's `started_at`/`completed_at`, or `started_at` twice when `completed_at` is None; the three legacy
cycle-summary stats become `cycle_summary`); serialization writes the stored shape (`_error` key, every
`legacy_meta` key verbatim, the stage keys) so persisted rows stay readable by every SQL reader, and a
decode → `compare_and_set_run_stage_reports` round trip (`citation_repair.py`) preserves `_resume`;
`journal.py::ReconciliationJournal.get_run_stage_execution` keeps reading `'_resume' in reports` from the raw
blob (task 7876 / task 3731 own that field); `stage_reports` keys are `StageId` members, never strings.

```python
# C5 — run streak declarations  (fm/reconciliation/run_streaks.py)   ε
@dataclass(frozen=True)
class RunStreak:
    name: str
    stage: StageId
    read: Callable[[StageReport], bool | None]   # True = bad cycle, False = good, None = no signal (stops the run)
    threshold: int
    escalation: ReconEscalation                  # S1/C8 registry member (category, severity, folds=True, identity)
    summary: Callable[[int, ReconciliationRun], str]   # renders the filing summary from the streak count
RUN_STREAKS: tuple[RunStreak, ...]               # today: stale task-count snapshot; growth-sweep failure
async def evaluate_run_streaks(streaks, journal, run: ReconciliationRun) -> list[StreakFiring]: ...
```

Invariants (C5): the window is the project's completed `RunType.full` runs other than `run` (today's filter,
written once); evaluation never raises (a journal read failure is logged and yields no firing); a streak fires
on every evaluation whose `StreakStep.count >= threshold` (sitting ruling OQ-1), so a resolved-but-persisting condition files
again on the first bad cycle after the 24-hour recently-resolved window
(`harness.py::_RESOLVED_RECURRENCE_WINDOW_SECONDS`), instead of never until a good cycle; the harness files each firing through
`policy_for(streak.escalation)` (S1/C1, S1/C8) and adds no level, severity or dedupe parameter of its own;
repeated firings fold onto the open record by the member's `folds=True`; a new alarm is one `RunStreak` value
and one `ReconEscalation` member.

```python
# C6 — recon task writes go through the interceptor   κ1, κ2
class ReconTaskWritePort(Protocol):               # TaskInterceptor satisfies it
    async def update_task(self, **kw) -> dict: ...
    async def set_task_status(self, **kw) -> dict: ...
    async def set_task_claimant(self, **kw) -> dict: ...
    async def submit_task(self, **kw) -> dict: ...

# An extension of S10 μ2's committed contract (task 9280): the frozen InterceptorCollaborators bundle gains one
# more provider-callable member, resolved by the interceptor at the end of its own construction — the pattern
# S10 already rules for the lifecycle-reset filer. Never a setter, never a post-construction assignment.
@dataclass(frozen=True)
class InterceptorCollaborators:                   # S10 μ2's bundle; κ1 adds the member below
    ...
    reconciler: Callable[[ReconTaskWritePort], TargetedReconciler] | None = None   # κ1 (lead ruling OQ-2 = S15/OQ-9 (a))
class TargetedReconciler:
    def __init__(self, memory, taskmaster, journal, config, buffer, *, task_writes: ReconTaskWritePort,
                 planned_episode_registry): ...
```

Invariants (C6): a `TargetedReconciler` always has its write gate; every reconciler write is classified once
(`write_failure_reason(resp) is None` means landed) and a refusal is a `run_actions` row, never a landed write;
the harness and its stages receive the port at construction; no module under `fm/reconciliation/` except
`consolidation_auto_executor.py` writes a task through the raw backend (guard test; the exemption is an INV-12
`Policy` entry, D10). S15 D12 consumes C6; S15 η → S12 κ1.

```python
# C7 — stage capabilities and prompt fragments   ι1, ι2
@dataclass(frozen=True)
class StageCapabilities: can_file_tasks: bool; can_escalate: bool; can_create_subtasks: bool
def capabilities_for(stage: StageId) -> StageCapabilities: ...   # derived from cli_stage_runner.STAGE{1,2,3}_DISALLOWED,
                                                                 # which stay in cli_stage_runner.py; stage_capabilities.py imports them
# fm/recon_state/recon_self_model.py (moved there by S19 β1, lead ruling OQ-12)
def render_dedup_anchor_rule() -> str: ...                       # rendered from FINGERPRINT_IDENTITY_FIELDS
# fm/reconciliation/prompts/fragments.py (ι2; prompts/__init__.py in ι1) — the fragment library; prompts/__init__.py keeps no names
def render_recon_report_tool_guidance(signatures: Mapping[str, inspect.Signature]) -> str: ...
    # pure; called by server/main.py::run_server with server.recon_report.get_recon_report_tool_signatures();
    # the rendered string is passed to ReconciliationHarness(…, report_tool_guidance=…)
# fm/memory_metadata.py — CHILD_KINDS (exists) and PARENT_ID_KEY (ι1 adds); server/grouped_read.py re-binds both from it
```

## Boundary-test sketch

| # | Side | Scenario | Pre | Post |
|---|---|---|---|---|
| 1 | driver | full run, all stages succeed | buffered events | one `runs` row `completed`; `stage_reports` keyed by `StageId`; watermark advanced; events `drained` by `run_id`; `runs.events_processed` equals the drained count, and was written before the first stage started (invariant 10) |
| 2 | driver | cancellation inside Stage 2 | full, events drained | row `failed`, `error.error_type == 'CancelledError'`, `failed_stage == task_knowledge_sync`; this run's events `buffered`; `CancelledError` propagates |
| 3 | driver | cancellation after `complete_run('completed')` committed | journal says completed | row stays `completed`; no restore; `run_end_skipped_after_durable_completion` logged |
| 4 | driver | shutdown cancellation between `start_run` and the first stage, backlog chunk | chunk ids given | after `_recover_stale_runs`, the chunk's events are `buffered` (no orphan drained rows) |
| 5 | driver | remediation, post-completion judge spawn raises | parent completed; remediation stages succeed | remediation row stays `completed`; no `recon_integrity_issue`; `post_completion_tail_failed` logged |
| 6 | driver | remediation, all findings fresh | scope-freshness pre-check returns all skipped | row `completed`; `stage_reports[StageId.integrity_check].agent_outcome == not_launched`; next cycle's persistence count includes it |
| 7 | driver | `AllAccountsCappedException` in Stage 1, each kind | — | row `failed`, `error.deferred is True`; events restored; run returned, nothing raised |
| 8 | journal | legacy row with `_error`, `_resume` and a synthetic `{'items_flagged': [...]}` | row written by today's code | `_row_to_run` returns typed run; `error` set; `_resume` in `legacy_meta`, not in `stage_reports`; synthetic entry a `StageReport` whose timestamps are the run's; a citation repair (decode, mutate a report, `compare_and_set_run_stage_reports`) leaves a blob that still carries `_error` and `_resume`; `get_run_stage_execution` still reports `resumed` |
| 9 | stage | `assemble_payload` called twice on two contexts | one stage instance per run (fixture builds two) | each `PayloadResult.stats` carries only its own run's counts; no attribute differs on the instances afterwards |
| 10 | stage↔server | a Stage-2 subprocess's `add_memory` call omits `agent_id` and `_causation_id`; a second passes the same values as the headers | MCP entry built by `_build_mcp_config(run)` | both land; each `write_ops` row carries `recon-stage-task_knowledge_sync` and the run id; `stats_verifier` attributes them; the `'recon-report'` entry of the same config carries the same two headers |
| 10a | stage↔server | a stage call passes an `agent_id` that differs from `STAGE_AGENT_ID_HEADER` (`add_memory`, `submit_task`, `get_entity`); another passes a `metadata['_causation_id']` that differs from `CAUSATION_ID_HEADER` (`add_memory`, `submit_task`) | headers set | each is refused: the caller reads `error_type == 'stage_identity_mismatch'`, the field, both values and the header name; no `write_ops` row, no memory, no ticket and no task is written; exactly one `recon_stage_identity_refused` log line per refused call; no `recon_stage_identity_override` event exists anywhere |
| 10b | stage↔server | Stage 2 calls `submit_task` with no `_causation_id` in its metadata | headers set | the creation-journal row the interceptor writes carries `causation_id` = the run-id header and `agent_id` = the stage identity; `read_run_creation_outcomes(run_id)` returns it |
| 10c | prompts | the three stage system prompts and the sweep files are rendered or read | — | each stage system prompt contains "The connection already identifies you and this run. Do not pass `agent_id` or `_causation_id`." and no instruction to send `agent_id`, `_causation_id` or a `recon-stage-*` id; `CLAUDE.md` §"Write-Tagging Convention", `.claude/commands/memory.md`, `docs/task-authoring.md` and the server's MCP instructions (`FUSED_MEMORY_INSTRUCTIONS`) state the reconciliation-stage exception and none names `reconciliation-stage-1` |
| 10d | stage↔server | a stage calls `search(agent_id='claude-task-7')` (the authoring-agent filter) | headers set | the search runs with that filter exactly as today; no refusal, no `recon_stage_identity_refused` line |
| 11 | server | a non-recon client sends no headers | — | identity and causation resolve exactly as today |
| 12 | streaks | snapshot misses ×4 then hit then ×3, threshold 3, real recon queue | journal of completed full runs | the filer is called at misses 3 and 4 of the first run and at miss 3 of the second; while the first record is open every call folds onto it (one record, dedupe count rising); a record resolved between misses 3 and 4 is suppressed at miss 4 and filed again by the first miss more than 24 h after the resolution (sitting ruling OQ-1); remediation runs ignored |
| 13 | targeted | interceptor refuses a hints write | gate returns `{'error': ..., 'error_type': ...}` | `run_actions` has `write_refused`; no `hints_attached` |
| 14 | targeted | knowledge search raises in `_on_task_done` | — | run `completed`; `run_actions` has `fail_open:knowledge_search` |
| 15 | stage | Stage 2 queues a briefing refresh | mismatch present | `submit_task` called on the write port; a curator ticket exists; raw `add_task` never called; the guard's only exemption is `consolidation_auto_executor.py`, and the INV-12 register reports it `Policy` |
| 16 | prompts | a denial-table entry changes | test-local table | the stage prompt's capability text follows; neither a prompt module nor `harness.py` imports `fused_memory.server`; the harness's stages receive the recon-report guidance string `server/main.py` rendered |
| 17 | driver | a remediation stage raises | parent full run completed | exactly one `recon_integrity_issue`, keyed on the remediation run id; the remediation row `failed`; the parent row stays `completed`; `post_completion_tail_failed` logged for the parent |

## Cross-PRD relationship

Edge notation: "X → Y" means X depends on Y (Y lands first).

| Other PRD / stream | Direction | Seam mechanism | Owner | Status / order |
|---|---|---|---|---|
| S14 `plans/review-all-s14-shared-leaves-prd.md` | consumes | `shared/streak.py::KeyedStreaks` (λ 6712) | S14 | done; ε uses it |
| S14 | consumes | `shared/agent_identity.py::recon_stage_agent_id`, `AgentIdentity` (μ 6721) | S14 | α → 6721 (`stats_verifier.py`); δ → 6721 |
| S10 `plans/review-all-s10-task-write-contract-metadata-modes-success-signal-and-registry-prd.md` | same file | `fm/reconciliation/harness.py`; last wave-2 task on it: S10 λ 9254 (λ 9254 → κ2 9251) | each | α → 9254 |
| S10 / S26 | same file | `stages/task_knowledge_sync.py`; last: S10 α 9180 (wave-2 order: S10 α 9180 → S26 μ 9203) | each | γ1 → 9180 |
| S26 `plans/review-all-s26-memory-service-write-path-and-graph-client-prd.md` | same file | `stages/memory_consolidator.py`; only S26 μ 9203 | each | γ1 → 9203 |
| S10 / S26 | same file | `fm/server/tools.py`; last: S10 μ2 9280 (after S26 λ, μ, ρ and S10 α, γ) | each | δ → 9280 |
| S10 / S26 | same file | `fm/server/main.py`; last: S10 μ2 9280 (S10 μ2 9280 → S10 κ1 9248 → S26 γ 9158) | each | ι1 → 9280; κ1 → 9280 |
| S10 | same file | `fused-memory/tests/_interceptor_helpers.py` (created by S10 μ1 9278) and `fused-memory/tests/test_auto_consolidation_e2e.py` (S10 μ1 9278, μ2 9280) | each | no edge onto 9278 for these test files (rebase-only, the lead's test-file rule); κ1 → 9280 (`server/main.py`, `middleware/task_interceptor.py`) and 9280 → 9278, so κ1 edits `_interceptor_helpers.py` after it exists |
| S10 | same file | `fm/middleware/task_interceptor.py`; last: S10 ν 9282 | each | δ → 9282 (δ passes the run-id header to `CreationAttribution.from_submission`; added by the OQ-3 ruling: 9282 has 4 more live ancestors than 9280, none on the merge-lane chain); κ1 reaches 9282 through δ and keeps its own edge |
| S10 | same file | `fm/reconciliation/targeted.py`; S10 α 9180, δ 9204, ο 9285, last ο | each | no edge: `targeted.py` vs S10 ο 9285 edits different symbols (ο rewrites task-status literal comparisons; κ1 the write gate), rebase-only, file locks order them; 9285 waits on 8528 (precedents: the S13 draft's row for S10 α, the S19 draft's D8 exception) |
| S10 | consumes | C5 `StatusCensusUnavailable` (S10 κ2, filed): the census code β1 moves into the driver already raises it | S10 | through 9254 |
| S3 | same file (tests only) | `fm/tests/test_stages.py`: S3 δ 9167, ε1 9188, ε2 9193 edit `_REVIEWED_STAGE_SAFE` | each | no edge; disjoint content, file locks (9188 and 9193 wait on 8528) |
| S1 (wave 3, concurrent) | consumes | C8 `reconciliation/recon_escalations.py::ReconEscalation` + `policy_for(member)` and C1 `FilingPolicy` (S1 ι1): ε declares one member per streak alarm and files through `policy_for` | S1 | ε → S1 ι1 (lead ruling OQ-7; S1 task id wired at decompose). S1 ω (S1's integration gate) → S12 ε, as the lead's cross-stream alignment has it; no other S1 leaf depends on an S12 leaf. S1 ι2 (which waits on 9285, hence on 8528) drops the growth alarm's stage-side filing in `task_knowledge_sync.py`: ε deletes it and files the alarm through C8 under `FilingPolicy.open_l1` (level 1, as today). β1 and S1 ι1 are unordered (both repoint `_escalate` calls in `harness.py`; whichever lands second rebases) |
| S13 (wave 3, concurrent) | consumes | S13 ι, the verbatim split of `stages/task_knowledge_sync.py` (`IntegrityCheck` → `stages/integrity_check.py`, the growth sweep and its streak escalation → `reconciliation/standing_decision_growth_sweep.py`, the reify briefing hook → `reconciliation/policies/briefing_refresh.py`) | S13 | S12 γ0, γ1, ε, κ2 → S13 ι (verbatim move first; lead ruling OQ-8 = S13/OQ-6 (a)); S13 ι takes no edge on any S12 task; S13 θ3 and S13 ε → S12 ι2; S13 ι edits `harness.py`'s `IntegrityCheck` import, rebase-only against α, β1, β2; γ1 reaches S13 α through S13 ι. Test files γ0 shares with S13, rebase-only: `fused-memory/tests/reconciliation/test_stage2_recon_report_channel.py` (S13 α; unordered) and `fused-memory/tests/reconciliation/test_task_knowledge_sync_citation_verify.py` (S13 ι; ordered before γ0) |
| S13 | produces (to) | C2 `StageCollaborators` (γ2); C3's headers on the recon-report entry (δ) and C3's refusal rule; C4 typed reports | S12 | S13 ν → S12 γ2 (a contract seam; both edit `stages/task_knowledge_sync.py`); S13 ν and S12 θ share only test files, so no edge (the lead's test-file rule); S13 ξ (the recon-report server reads the headers) → S12 δ (lead ruling OQ-10). S13 ξ also calls C3's predicate from `server/recon_report.py::_report_run_id`, widening `StageIdentityRefusal.field` by `'run_id'` (the lead's ruling: one predicate, one error shape; C3's comment names that caller). Under the OQ-3 ruling's cross-stream rule S13 ξ REFUSES a `run_id` argument that differs from `CAUSATION_ID_HEADER` (a structured error naming both values, one log line per refusal; no override event), and removes the prompt's "echo run_id" line in the same change; it may call C3's `stage_identity_refusal` (its choice). C3's header names, `stage_mcp_headers` and every S12 task label are unchanged. S13's edge sweeps take their agent id from S14 μ's `recon_stage_agent_id`, not from C3 |
| S13 | same files | `models/reconciliation.py` (S13 ε adds `FindingCategory`; S12 α, ζ), `cli_stage_runner.py` (S13 ε), `recon_self_model.py` and `prompts/stage1.py`–`stage3.py` (S13 ε, θ3, κ2; S12 δ, which now edits `stage1.py`, `stage2.py` and `recon_self_model.py` for the sweep, ι1, ι2), `prompts/fragments.py` (created by S12 ι2; S13 ε renders the category vocabulary there) | each | lead's per-file order for `models/reconciliation.py`, the prompts and the self-model; S13 ε → S12 ι2 for `prompts/fragments.py`; S13's prompt tasks after θ3 reach S12 δ through ι2 |
| S15 (wave 3, concurrent) | produces (to) / same files | C3 header identity and refusal, which S15's tool families reach through the per-tool resolver call (S15 D6); C6's provider member | S12 | S15 D12 consumes C6; S15 η → S12 κ1 (`targeted.py`, `main.py`; lead ruling OQ-2 = S15/OQ-9 (a)). S12 δ → S15 α (δ edits `_resolve_identity`/`_extract_causation` where S15 α moved them, `server/tool_context.py`). Lead ruling OQ-11 (a1): δ also passes `ctx` at the 17 `_extract_causation` call sites in `server/tools.py`, which S15 ε1–ε5's verbatim moves carry; S15 δ → S12 δ and S15 ε1 → S12 δ. S15 δ (gate protocol, `split_write_controls(…, header_causation_id=…)`) takes the causation id from S12's resolver, which has already refused a differing `_causation_id`, so `split_write_controls` never decides precedence: S15 C5's "a given `header_causation_id` wins over `_causation_id`" reads "fills an absent `_causation_id`", and S15's boundary row 16 drops its "wins" assertion. S15 λ vs δ and κ1 on `task_interceptor.py`: lead's per-file order |
| S19 (wave 3, concurrent) | consumes | C2 `fused_memory.recon_state` (S19 α moves `journal.py`, `event_buffer.py`; S19 β1 moves `recon_self_model.py`) | S19 | lead ruling OQ-12: S12 α, ζ, κ1 → S19 α and δ → S19 β1 (δ now edits `recon_self_model.py` for the sweep; ι1 reaches S19 β1 through δ), with Files at the `recon_state/` paths (α → S19 α covers the whole run chain); S19 β2, which moves other modules and repoints imports in `harness.py` and the stages, is unordered, rebase-only. S19's row "S12 ι1 → β1" becomes "S12 δ → β1" |
| S19 | produces (to) | ι1 removes both `prompts`→`server` imports (`prompts/__init__.py`'s function-local `server.recon_report` import, `prompts/stage1.py`'s `server.grouped_read` import) | S12 | S19's subpackage-DAG test → S12 ι1. S19 δ → S12 ι1: S19 δ moves `server/grouped_read.py`, whose `CHILD_KINDS`/`PARENT_ID_KEY` re-bind ι1 makes, and carries it (S19's draft holds that edge on ι1 keeping the file, which it does). S19 δ's bullets "the recon prompt builder takes the recon_report tool specs as data; `server/main.py` passes them; the deferred import goes" are the same work as ι1: one owner (ι1); S19 ε → S12 ι1 (lead ruling S19/Q-C4: S19 δ drops the cut) |
| S16 (wave 3, concurrent) | same files | S16 ε2a rewrites `ReconciliationHarness._select_tier`, deletes the `opus_model` field `BacklogIterator.run` reads into `TierConfig`, and edits `judge.py`, `cli_stage_runner.py`; S16 ε1 edits `reconciliation/judge.py` (S12 α too); S16 κ's prose sweep edits `CLAUDE.md` (other lines than δ's §"Write-Tagging Convention") | each | S16 ε2a → S12 β1 (ε2a rebases onto `drive`); β1 never depends on ε2a, which waits on 8528, 7720 and 5040 through 9188. `RunRequest.tier` carries whatever `_select_tier` returns after ε2a. S16 ε1 → S12 α (`judge.py`). S16 κ → S12 δ (`CLAUDE.md`; the lead's ruling, already implied through S16 β1 → β2 → S15 ε1 → S12 δ). No S12 task depends on an S16 task |
| S2 (wave 3, concurrent) | same files | S2 γ2 and γ3 (the fingerprint retirement: readers, then the field) and S2 κ (renames the `infra_dedupe_*` attributes, parked behind 8528) edit `reconciliation/harness.py`; κ also edits `task_count_snapshot_cadence.py` (ε deletes its streak quartet) | lead | the lead orders S2 γ2 against S12's tasks on `harness.py` at decompose, as S2's draft states (γ2 changes no write, so either order is safe); S2 γ3 and κ follow γ2 on S2's own chain. No S12 task depends on an S2 task |
| S24 (wave 3, concurrent) and filed tasks | same files (docs) | δ's sweep edits `CLAUDE.md` §"Write-Tagging Convention" (S24's lease task edits §"Orchestrator Fleet Redeploy"; 22 live filed tasks declare `CLAUDE.md`), `.claude/commands/memory.md` (task 3139 declares it) and `docs/task-authoring.md` §"Write-tagging convention" (wave-2 tasks 9199 and 9244 edit other sections; 58 live tasks declare it), measured 2026-11-12 with `find_tasks_by_metadata(key="files", value=<path>)` | each | S16 κ → S12 δ on `CLAUDE.md` (S16 row). Otherwise no edge: one-line edits in a section no other task edits; file locks order them |
| S20 (wave 3, concurrent) | same files | `reconciliation/judge.py` (S12 α; S20 ξ2), plus test files (`test_judge.py`, `test_stage_agent_outcome.py`, `test_recon_amend_tool_advertisement.py`, `test_stages.py`: S12 α, γ0, ι2 and others; S20 ξ2, ο1) | lead | `judge.py`: S20's rule (the plan decomposed second depends on the other's last leaf on the file), so the lead orders S12 α and S20 ξ2 at decompose; the test files are rebase-only (the lead's test-file rule) |
| `plans/recon-reliability-prd.md` §8.4 (self-model) | extends | `recon_self_model.py` rendered sections | this PRD (ι1) | the pending "ξ prompt cutover" is completed by ι1 |

Every other live task on these files (§Live tasks) is ordered by the scheduler's file locks; edges are wired
only where the content interacts.

## Live tasks (contract §8)

- **6695** (pending, low) — the residual of `fk-0485729893fe` exactly. Kept: β1 → 6695; the lead stamps the key
  on it. D4 keeps its fix.
- **5912** (pending, low) — "route `run_full_cycle`'s cancellation `_error` through `_run_failure_record`".
  Subsumed by α (`RunError.from_exception`) and β1: cancelled with `x_acceptance_reason`.
- **6694** (deferred, low) — a cycle-timeout cancel journaled `interrupted`. Its own deferral
  (`metadata.deferral`, `carried_by` 5514) says to close it as found on main citing 5514's landing commit:
  the lead closes it `done` with `done_provenance` `found_on_main` and 5514's merge commit. Not subsumed by
  this PRD.
- **7876** (pending, low) — removes the legacy `interrupted` status and resume leftovers; it waits on a
  live-store observation (zero `interrupted` rows). α → 7876.
- **7877** (pending, low) — is `_ensure_stage2_cycle_summary`'s degraded-synth arm still reachable. ζ → 7877
  (its answer may delete `DEGRADED_SYNTH`).
- **6783** (pending, low) — partition `harness.py` starting with `BacklogIterator`. Subsumed by η1/η2:
  cancelled.
- **7343** (pending, low) — public seam for `BaseStage`'s escalation endpoint and queue. Subsumed by γ2:
  cancelled.
- **8835** (pending, low) — test seam for the harness escalation URL in `test_stages.py`. Subsumed by γ2 (the
  endpoint is a constructor collaborator): cancelled.
- **7604** (pending, low) — drive tree-wiring tests through the constructor; it asks for the tests spying on
  `_configure_consolidator` / `_configure_task_sync` to move to constructor seams, and γ2 deletes those
  helpers. Subsumed by γ2 (θ finishes): cancelled.
- **8144** (pending, medium) — extracts `_sweep_cancelled_descendants` and its "not wired; skipping" guard
  into the new `reconciliation/cancel_backstop.py`, where κ1 then deletes that guard. κ1 → 8144, and
  `cancel_backstop.py` is in κ1's Files.
- **6846** (pending, low) — extract `IntegrityCheck` to `stages/integrity_check.py`. Not subsumed by this PRD;
  S13 ι cancels it and does the move, and γ1 and γ2 edit `IntegrityCheck` at `stages/integrity_check.py`
  (lead ruling OQ-8; ζ does not touch `IntegrityCheck`, which writes no cycle summary).
- **6895** (pending, low; found by the 2026-11-12 dedup re-run for δ's widened scope) — "enforce
  `metadata._causation_id` on recon-stage `submit_task` calls so every creation-journal row is attributed to its
  run", a boundary guard rejecting a recon-stage submission that omits it. Superseded by δ under the OQ-3
  ruling: the run id is bound on the stage's connection and task filing reads it from the header, so a stage
  submission cannot omit it, and the prompt now tells the stage NOT to send it (a guard requiring it would
  contradict the ruling). Cancelled with `x_acceptance_reason`.
- **Priority inversions** (contract §4; recorded, no override, as the S11 FILED row does): the high leaves β2,
  δ and ι1 wait, through the run chain, on the low tasks 7876 (via α) and 6695 (via β1). 7877 is not an
  ancestor of a high leaf (only ζ depends on it).
- Same-file only, file locks order them: 3991, 4875, 5072, 5154, 5270, 6084, 6234, 6634, 6678, 7469, 7894,
  8015, 8385, 8604 and others listed by `find_tasks_by_metadata(key="files", value=<path>)`.

## Decomposition plan

Every task is `task_kind="normal"`. `priority` is the highest severity of the keys it closes; **β2, δ and ι1
are `high`** and become dependencies of trigger-chain gate 6698 at decompose. Tasks that share a non-test
file are serialized by real edges, checked file by file against every Files list below: the run chain on
`harness.py` (α, β1, β2, γ1, γ2, δ, ι1, ε, ζ, κ1, κ2, η1, η2, θ); on `stages/task_knowledge_sync.py` the
landing order γ1, γ2, ε, ζ, κ2, ι2; on `stages/integrity_check.py` γ1, γ2; on `stages/base.py` γ1, γ2, δ, ι1,
κ2; on `prompts/stage1.py`, `prompts/stage2.py` and `recon_state/recon_self_model.py` δ, ι1, ι2 (ι2 does not
edit the self-model); on `middleware/task_interceptor.py` δ, κ1; on `server/main.py` ι1, κ1, κ2. Tasks that
share only test files are not chained (the lead's test-file rule, intra-plan too): `fused-memory/tests/test_stages.py`,
`test_harness.py` and the other shared test modules are rebase-only, so γ0 takes no edge onto β2 and θ none
onto ι2. One test-only edge is kept and named for the lead: γ1 → γ0, because γ1's declared work edits
`fused-memory/tests/reconciliation/stage_builders.py`, the helper γ0 creates, in place of the 34 modules
(γ2 then edits it after γ1 on a real edge). ω, the integration gate, depends on θ and ι2. Files that a sibling move creates
(the lead's rulings S12/OQ-8 and S12/OQ-12) are named at the new paths and marked "created by …" or "moved
there by …". Nothing is held. No follow-up task is filed later (the OQ-3 ruling adds none). Every task carries `metadata.x_finding_key` (its keys),
`x_finding_run = "review-all-dark_factory-20261008"`, `source = "review-all"`, and `metadata.files` = its
Files list.

**α — Typed reconciliation run record: StageId-keyed reports, a RunError field, the journal as sole decoder**
- Size and keys: ~1,000 LOC. Key `fk-377a7036ac7f`. Priority medium.
- Files: `fused-memory/src/fused_memory/models/reconciliation.py`, `fused-memory/src/fused_memory/recon_state/journal.py` (moved there by S19 α), `fused-memory/src/fused_memory/reconciliation/harness.py`, `fused-memory/src/fused_memory/reconciliation/stats_verifier.py`, `fused-memory/src/fused_memory/reconciliation/judge.py`, `fused-memory/src/fused_memory/reconciliation/citation_repair.py`, `fused-memory/src/fused_memory/reconciliation/cross_run_citation_scan.py`, `fused-memory/src/fused_memory/reconciliation/episode_window.py`, `fused-memory/tests/test_journal.py`, `fused-memory/tests/test_harness.py`, `fused-memory/tests/test_stats_verifier.py`, `fused-memory/tests/test_judge.py`, `fused-memory/tests/reconciliation/test_run_record.py` (new)
- Deps: 7876, 6721, 9254; S19 α (`recon_state/journal.py`; S19 task id wired at decompose; lead ruling OQ-12).
- Work: C4 except `cycle_summary` (ζ). Every `'_error'` write becomes `run.error = RunError…`; the remediation
  short-circuit builds a `StageReport` (D5) with the run's timestamps; legacy `_`-prefixed keys other than
  `_error` decode into `legacy_meta` and serialize back verbatim; the 4 `isinstance(s3_report, dict)` branches,
  the judge's non-report pass-through, `stats_verifier`'s StageId-parse skip and the three `isinstance(report,
  StageReport)` guards go. The judge prompt shows `run.error` as its own field. `extract_snapshot_written` and
  `_extract_growth_sweep_failed` keep working on typed reports and are deleted by ε. `get_run_stage_execution`'s
  raw `'_resume' in reports` read is untouched (task 7876 / task 3731 own it).
- Deletes: the `'_error'` dict literals (7) and their pop/read; the shape branches above; the
  `dict[str, StageReport | dict]` annotation.
- Signal: boundary row 8 in CI (including the citation-repair round trip that keeps `_resume`); after deploy,
  `get_run` on the live journal's newest failed run returns `error.error_type` set and `stage_reports` keys
  that are `StageId` members.
- Consumer: β1 (the driver writes `run.error`), judge, `stats_verifier`, every journal reader.
- G7: INV-2 holds (the error record is typed and names the stage); INV-9 holds (one decoder).

**β1 — One reconciliation run driver for full and backlog-chunk runs; the run row before any drain**
- Size and keys: ~1,100 LOC. Key `fk-0485729893fe`; builds C1 for `fk-8621784d79d3` (closed by β2). Priority medium.
- Files: `fused-memory/src/fused_memory/reconciliation/run_context.py` (new), `fused-memory/src/fused_memory/reconciliation/harness.py`, `fused-memory/src/fused_memory/recon_state/journal.py` (moved there by S19 α), `fused-memory/tests/test_harness.py`, `fused-memory/tests/test_journal.py`, `fused-memory/tests/reconciliation/test_throughput.py`, `fused-memory/tests/test_event_buffer.py`, `fused-memory/tests/reconciliation/test_run_driver.py` (new)
- Deps: α, 6695.
- Work: C1 for `FULL` and `BACKLOG_CHUNK` (D1–D4); `TierConfig` moves from `harness.py` into
  `run_context.py` (harness imports it) and `TreeSnapshot` is built there; `RunContext` type (C2) built by
  the driver and passed to stages as today's `run_id`/`model` until γ1; `ReconciliationJournal.set_events_processed`
  and invariants 0, 1 and 10 (the drain is the first statement inside the ladder's `try`, the count written right
  after it, `len(ids)` for a chunk; the lifecycle inside `_active_runs.track`, the post-completion tail before
  the `finally`); `run_full_cycle` becomes a wrapper; `BacklogIterator._run_chunk` and
  `_run_final_consolidation` build requests and stop draining and restoring; the iterator's deadline
  `except TimeoutError` restore goes (invariant 9).
- Deletes: `BacklogIterator`'s `drain_by_ids` + restore pair and its deadline restore; `run_full_cycle`'s
  inline lifecycle (moved into `drive`).
- Signal: boundary rows 1–4 and 7 for full and backlog chunk (row 1 asserts `events_processed`); after
  deploy, read-only sqlite over `data/reconciliation/reconciliation.db`: `SELECT count(*) FROM event_buffer e
  LEFT JOIN runs r ON r.id = e.drained_by_run_id WHERE e.status='drained' AND r.id IS NULL` is 0 at every
  sample (drained rows are deleted after an hour, so sample within that window), and each full run sampled
  that way has `events_processed` equal to its count of `drained_by_run_id` rows.
- Consumer: β2, γ1, ε, `BacklogIterator`.
- G7: INV-11 holds (no unattributed drain); INV-6 `status-matches-liveness` holds (the run row is written
  before ownership of events is taken).

**β2 — Remediation is RunKind.remediation on the one driver; the second run lifecycle is deleted**
- Size and keys: ~1,300 LOC (≈ 800 deleted). Keys `fk-8621784d79d3`, `fk-2571f7726184`. Priority **high**.
- Files: `fused-memory/src/fused_memory/reconciliation/harness.py`, `fused-memory/src/fused_memory/reconciliation/run_context.py`, `fused-memory/tests/test_harness.py`, `fused-memory/tests/reconciliation/test_remediation_tail_offload.py`, `fused-memory/tests/reconciliation/test_escalation_archive.py`, `fused-memory/tests/reconciliation/test_stage1.py`, `fused-memory/tests/reconciliation/test_run_driver.py`
- Deps: β1.
- Work: the remediation policy row; scope freshness as the pre-step (D5); the persistence-gated escalation as
  the kind tail; `_run_remediation_pass` a one-call wrapper; D2/D3 for remediation (durable completion final,
  re-raise into the parent's tail); `_maybe_remediate`'s `try` narrowed to its own orchestration
  (partitioning, suppression, the deferral gate), with the `drive(RunRequest.remediation(…))` call after it
  (invariant 11). Tests that called `_run_remediation_pass` call `drive(RunRequest.remediation(…))`.
- Deletes: `_run_remediation_pass`'s own `ReconciliationRun(…)`, stage loop, pre-Stage-3 flush call,
  `complete_run`/verify/persist/judge sequence, three except arms and `finally` (≈ 800 lines); the
  byte-identical `AllAccountsCappedException` literal; the double filing of a remediation failure
  (`_maybe_remediate`'s `'Remediation orchestration failed'` arm keeps only the orchestration it still wraps).
- Signal: boundary rows 2, 5, 6, 7 and 17 for remediation in CI; after deploy, `journalctl -u fused-memory` shows
  `reconciliation.remediation_started` / `reconciliation.remediation_completed` pairs whose run ids read
  `completed` in `runs` (read-only sqlite), now emitted by the driver.
- Consumer: `_maybe_remediate`, ε (one tail), η2 (the finding gate moves).
- G7: INV-5 `no-lockstep-duplication` holds (the second lifecycle is gone); INV-11 holds.

**γ0 — Reconciliation stage tests build and run stages through one test helper**
- Size and keys: ~1,500 LOC of uniform substitution. Contributes to `fk-63b65973bd33` and `fk-0cce86f5dbfe`
  (the test side of both); closes none. Priority medium.
- Files: `fused-memory/tests/reconciliation/stage_builders.py` (new), `fused-memory/tests/reconciliation/consolidator_fixtures.py`, `fused-memory/tests/reconciliation/test_assemble_payload_snapshot_filter.py`, `fused-memory/tests/reconciliation/test_base_stage_cutover.py`, `fused-memory/tests/reconciliation/test_cutover_end_to_end.py`, `fused-memory/tests/reconciliation/test_episode_freshness_cursor.py`, `fused-memory/tests/reconciliation/test_flag_dedup_project_qualified_marker.py`, `fused-memory/tests/reconciliation/test_integrity_check_dangling_citation_scan.py`, `fused-memory/tests/reconciliation/test_known_projects_section.py`, `fused-memory/tests/reconciliation/test_live_workflow_section.py`, `fused-memory/tests/reconciliation/test_orphaned_recon_escalation_sweep.py`, `fused-memory/tests/reconciliation/test_overreach_flags_section.py`, `fused-memory/tests/reconciliation/test_recon_stage_escalation_read_boundary.py`, `fused-memory/tests/reconciliation/test_stage1_assembled_budget.py`, `fused-memory/tests/reconciliation/test_stage1_auto_consolidation_wiring.py`, `fused-memory/tests/reconciliation/test_stage1_child_memories.py`, `fused-memory/tests/reconciliation/test_stage1_payload_section_parity.py`, `fused-memory/tests/reconciliation/test_stage1.py`, `fused-memory/tests/reconciliation/test_stage1_run_failure_isolation.py`, `fused-memory/tests/reconciliation/test_stage1_unlanded_path_wiring.py`, `fused-memory/tests/reconciliation/test_stage2_flag_carry_forward.py`, `fused-memory/tests/reconciliation/test_stage2_recon_report_channel.py`, `fused-memory/tests/reconciliation/test_stage_agent_outcome.py`, `fused-memory/tests/reconciliation/test_standing_decision_e2e.py`, `fused-memory/tests/reconciliation/test_standing_decision_growth_sweep.py`, `fused-memory/tests/reconciliation/test_task_count_snapshot_cadence.py`, `fused-memory/tests/reconciliation/test_task_knowledge_sync_citation_verify.py`, `fused-memory/tests/test_auto_consolidation_e2e.py`, `fused-memory/tests/test_delivered_checks_e2e.py`, `fused-memory/tests/test_harness.py`, `fused-memory/tests/test_recon_amend_tool_advertisement.py`, `fused-memory/tests/test_recon_gate_closure_guidance.py`, `fused-memory/tests/test_stage2_stats_audit.py`, `fused-memory/tests/test_stage2_tasks_created_accounting.py`, `fused-memory/tests/test_stages.py`
- On the size: 35 files, crossing the >15-file trigger deliberately: a uniform substitution (the S10 μ1 9278
  precedent, one interceptor test factory over 49 files). The 34 modules are the union, measured at
  `1f240e6da2` and unchanged at `1b106418da` (of the union only `test_harness.py` and `test_stages.py` changed
  since, and their site counts are unchanged), of the test modules that reach a stage by `await stage.run(` (20 files, about 215 sites),
  direct stage construction (18 files, 149 sites), harness-set attribute writes (22 files),
  `._escalation_(queue|url) =` writes (10 files) or `.assemble_payload(` calls (11 files).
- Deps: S13 ι (`stage_builders.py` imports `IntegrityCheck`, which the split moves; S13 task id wired at
  decompose; lead ruling OQ-8). No other: every file γ0 shares with β2 (`test_harness.py`, `test_stages.py`,
  `test_stage1.py`) or with S10 μ1 9278 and μ2 9280 (`test_auto_consolidation_e2e.py`) is a test file,
  rebase-only under the lead's test-file rule.
- Work: `build_stage(stage_cls, **inputs)` constructs a stage and applies today's post-construction
  assignments (harness-set attributes, escalation endpoint and queue) inside the helper; `run_stage(stage,
  events, watermark, prior_reports, *, run_id, model=None)` calls today's `run`; `assemble(stage, …)` calls
  today's `assemble_payload`. Every listed module calls them instead of constructing, assigning or running a
  stage itself. No production file changes; behaviour and assertions are unchanged.
- Deletes: the per-module stage construction and attribute-assignment boilerplate (the substitution sites
  above).
- Signal: the suite passes unchanged in CI; re-running the measurement rule above finds direct stage
  construction, stage attribute assignment and `_current_run_id` writes only in `stage_builders.py` (γ0's
  commit records the counts before and after).
- Consumer: γ1 and γ2 (each then changes `stage_builders.py` instead of 34 modules).
- G7: tests stance (one public seam for constructing and running a stage); INV-9 holds.

**γ1 — Stage payload interface: assemble_payload(…, run) returns PayloadResult; the per-run stashes go**
- Size and keys: ~1,100 LOC. Key `fk-0cce86f5dbfe`. Priority medium.
- Files: `fused-memory/src/fused_memory/reconciliation/run_context.py`, `fused-memory/src/fused_memory/reconciliation/stages/base.py`, `fused-memory/src/fused_memory/reconciliation/stages/memory_consolidator.py`, `fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py`, `fused-memory/src/fused_memory/reconciliation/stages/integrity_check.py` (created by S13 ι), `fused-memory/src/fused_memory/reconciliation/harness.py`, `fused-memory/tests/reconciliation/stage_builders.py`, `fused-memory/tests/test_stages.py`, `fused-memory/tests/test_harness.py`, `fused-memory/tests/reconciliation/test_stage1.py`, `fused-memory/tests/reconciliation/test_stage2_recon_report_channel.py`, `fused-memory/tests/reconciliation/test_task_knowledge_sync_citation_verify.py`, `fused-memory/tests/reconciliation/test_assemble_payload_snapshot_filter.py`, `fused-memory/tests/reconciliation/test_episode_freshness_cursor.py`, `fused-memory/tests/reconciliation/test_live_workflow_section.py`, `fused-memory/tests/reconciliation/test_task_count_snapshot_cadence.py`
- On the size: 16 files; after γ0 the 10 test files are `stage_builders.py` plus the 9 modules that assert
  one of the 10 stash attributes by name (rule: `git grep -l` of the 10 attribute names in
  `fused-memory/tests`, measured at `1b106418da`).
- Deps: β2 (`harness.py`, `run_context.py`), γ0 (`stage_builders.py`, which γ0 creates and γ1 edits), 9180,
  9203; S13 ι (`stages/task_knowledge_sync.py` split, `stages/integrity_check.py`; S13 task id wired at
  decompose; lead ruling OQ-8).
- Work: C2's `run`/`assemble_payload`/`after_report` (D6). `TaskKnowledgeSync`'s 6 stashes and
  `MemoryConsolidator`'s 4 become `PayloadResult.stats` / `carry`; the `run()` overrides become
  `after_report`; the `RuntimeError` telling tests to set `stage._current_run_id` goes.
- Deletes: the 10 stash attributes and their resets, writes and read-backs (47 references); the "reviewer
  finding: robustness" reset comments; the 6 test assignments to `_current_run_id`.
- Signal: boundary row 9 in CI; after deploy, a live cycle's Stage-1 and Stage-2 reports carry
  `stage1_fetch_degraded`, `stale_missing_run_id_markers` and `rescued_in_window_markers` (journal read).
- Consumer: γ2, δ, ε, ζ.
- G7: INV-5 holds; INV-9 holds (the run id has one home, `RunContext`).

**γ2 — Reconciliation stages constructed with their collaborators and per-cycle inputs**
- Size and keys: ~1,400 LOC. Key `fk-63b65973bd33`; contributes to `fk-16baf39b1db6`. Priority medium.
- Files: `fused-memory/src/fused_memory/reconciliation/stages/base.py`, `fused-memory/src/fused_memory/reconciliation/stages/memory_consolidator.py`, `fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py`, `fused-memory/src/fused_memory/reconciliation/stages/integrity_check.py` (created by S13 ι), `fused-memory/src/fused_memory/reconciliation/harness.py`, `fused-memory/tests/reconciliation/stage_builders.py`, `fused-memory/tests/reconciliation/consolidator_fixtures.py`, `fused-memory/tests/test_stages.py`, `fused-memory/tests/test_harness.py`, `fused-memory/tests/reconciliation/test_stage1.py`, `fused-memory/tests/reconciliation/test_base_stage_cutover.py`, `fused-memory/tests/reconciliation/test_overreach_flags_section.py`
- On the size: 12 files; after γ0 the test files are `stage_builders.py`, `consolidator_fixtures.py` and the 5
  modules that name `_configure_consolidator`, `_configure_task_sync`, `_propagate_escalation_queue` or
  `_make_stages` (rule: `git grep -l` of those four names in `fused-memory/tests`, measured at `1b106418da`).
- Deps: γ1.
- Work: C2's `StageCollaborators`, `ReconReportEndpoint`, `EscalationEndpoint` and `Stage{1,2,3}Inputs`;
  `_make_stages(scope, inputs)` builds stages with them; `build_stage` in `stage_builders.py` passes
  collaborators and inputs instead of assigning attributes; the harness takes an injectable `stage_factory`
  (the public seam the tests use instead of `harness._make_stages` / `harness.stages`).
- Deletes: `_configure_consolidator`, `_configure_task_sync`, `_propagate_escalation_queue`, the 17
  "set by harness" class attributes and their comments, `BaseStage._escalation_url` / `_escalation_queue` /
  `known_projects` post-construction writes.
- Signal: `grep -n 'set by harness' fused-memory/src/fused_memory/reconciliation/stages/` returns nothing;
  boundary row 9; a live cycle's stage reports are unchanged in shape (journal read).
- Consumer: δ, ι1, κ2 (collaborators), θ.
- G7: INV-5 holds (the inputs have one declaration); INV-9 holds.

**δ — Recon stage identity and run id bound on the MCP transport; a differing value refused; the stage instructions swept; fused-memory URL from server config**
- Size and keys: ~1,000 LOC (the transport and refusal ~700, task filing reading the header ~100, the
  instruction sweep ~100–150 lines of prompt and doc text, estimated in S12.md §S12-2, not measured). Keys
  `fk-a3598aec7dca`, `fk-d2c7f1b902ac`. Priority **high**.
- Files: `fused-memory/src/fused_memory/reconciliation/run_context.py`, `fused-memory/src/fused_memory/reconciliation/stages/base.py`, `fused-memory/src/fused_memory/reconciliation/harness.py`, `fused-memory/src/fused_memory/server/tool_context.py` (created by S15 α), `fused-memory/src/fused_memory/server/tools.py`, `fused-memory/src/fused_memory/reconciliation/stats_verifier.py`, `fused-memory/src/fused_memory/middleware/task_creation_journal.py`, `fused-memory/src/fused_memory/middleware/task_interceptor.py`, `fused-memory/src/fused_memory/reconciliation/prompts/stage1.py`, `fused-memory/src/fused_memory/reconciliation/prompts/stage2.py`, `fused-memory/src/fused_memory/recon_state/recon_self_model.py` (moved there by S19 β1), `CLAUDE.md`, `.claude/commands/memory.md`, `docs/task-authoring.md`, `fused-memory/tests/reconciliation/test_base_find_fused_memory_server.py` (deleted), `fused-memory/tests/reconciliation/test_base_stage_cutover.py`, `fused-memory/tests/test_stages.py`, `fused-memory/tests/test_stats_verifier.py`, `fused-memory/tests/test_task_creation_journal.py`, `fused-memory/tests/server/test_recon_stage_transport_identity.py` (new), `fused-memory/tests/reconciliation/test_stage_identity_instructions.py` (new)
- On the size: 21 files, over the band's 10–12, deliberately: the ruling puts the sweep and task filing's
  header read in this task and in the refusal's commit ("no new follow-up task"), because a stage reading
  the old "always pass `agent_id`" text would be refused the moment the refusal lands. The seven sweep sites (six files, plus
  the instructions string in `server/tools.py`, already in the list) each take a few lines of text.
- Deps: γ2, 6721, 9280, 9282 (`middleware/task_interceptor.py`, the OQ-3 ruling's added edge); S15 α
  (`_resolve_identity` and `_extract_causation` in `server/tool_context.py`); S19 β1
  (`recon_state/recon_self_model.py`; sibling task ids wired at decompose). Not S15 δ (gate protocol):
  S15 δ and S15 ε1 → S12 δ instead (lead ruling OQ-11 (a1)).
- Work: C3 and D7/D7a. `_build_mcp_config(run)` emits the fused-memory http entry with headers, its URL
  built with `shared.mcp_post.mcp_endpoint_url` from `config.server` (no new `ServerConfig` property, no
  trailing-slash literal), and puts the same `stage_mcp_headers(run)` on the `'recon-report'` entry (lead
  ruling OQ-10). `stage_identity_refusal` is the one predicate; `_resolve_identity` and `_extract_causation`
  supply the header's value for an absent or equal argument and refuse a differing one (never on `search`'s `agent_id` filter, D7), the caller reading
  `StageIdentityRefusal.as_error()` and the server writing one `recon_stage_identity_refused` line per
  refusal (the 17 `_extract_causation` call sites in `server/tools.py` pass the `ctx` they already hold; S15
  ε1–ε5's verbatim moves carry those edits, and S15's gate-protocol task takes the causation id from this
  resolver). `submit_task` resolves the run id the same way and the interceptor passes it to
  `CreationAttribution.from_submission(…, bound_causation_id=…)`. `stats_verifier` reads the stage
  identity through `recon_stage_agent_id`. The sweep, in the same commit: `stages/base.py`'s
  `recon_context` keeps `run_id`, `stage` and `project_id` for recon-report calls and replaces its three
  `agent_id` / `_causation_id` lines with "The connection already identifies you and this run. Do not pass
  `agent_id` or `_causation_id`."; `prompts/stage1.py`'s `propose_consolidation(…)` example drops its
  `agent_id=` argument (keeping the real `run_id=`); `prompts/stage2.py`'s "Creating Tasks" drops the
  `"_causation_id": <run_id>` metadata and its sentence; `recon_self_model.py`'s task-creation accounting
  text says the connection carries the run id; `CLAUDE.md` §"Write-Tagging Convention", `.claude/commands/memory.md`
  (beside "Always include `project_id` and `agent_id`") and `docs/task-authoring.md` §"Write-tagging
  convention" each gain the one-line exception, and `server/tools.py::FUSED_MEMORY_INSTRUCTIONS`' convention
  line "Include agent_id for attribution …" gains "a reconciliation stage omits agent_id and _causation_id:
  its connection already identifies it" (that phrase; the sidecar greps it) that reconciliation stages pass neither `agent_id` nor
  `_causation_id` because their connection identifies them, and the two docs that cite
  `"reconciliation-stage-1"` drop it. Nothing else changes: every other client keeps passing `agent_id`.
- Deletes: `_find_fused_memory_server` (and the hard-coded `/home/leo/src/dark-factory/.mcp.json`, the cwd
  candidate and the stdio fallback) and its test module; the three `recon_context` instruction lines; the
  Stage-1 `agent_id='recon-stage-memory_consolidator'` example argument; the Stage-2 `_causation_id`
  metadata instruction; the self-model's "ONLY when that `submit_task` call carried" sentence; the
  `"reconciliation-stage-1"` example in `CLAUDE.md` and `docs/task-authoring.md`.
- Signal: boundary rows 10, 10a–10d and 11; after deploy, read-only sqlite over the write journal: every
  `write_ops` row with `causation_id` = the newest completed full run's id carries
  `agent_id LIKE 'recon-stage-%'`, that run's Stage-2 creation-journal rows (when it filed a task) carry its
  run id, and the `stats_verifier: … matched no known stage` INFO line does not appear for that run;
  `journalctl -u fused-memory` shows no `recon_stage_identity_refused` line for that run, or each one names
  both values.
- Consumer: `stats_verifier`, the creation journal Stage 2's counter repair reads, every `recon-stage-` gate
  (S14 μ parses it), S13 ξ (reads `CAUSATION_ID_HEADER` under the same rule), S15's tool families and gate
  protocol, every stage session (the instructions).
- G7: INV-1 `contracts-machine-checked` holds (the identity contract is a header the server reads and
  enforces, not prose); INV-3 `corroborate-before-acting` not engaged; INV-23
  `one-decision-partition-per-area` holds (D13: a decidable fact leaves the LLM).

**ι1 — Stage prompts render capabilities and the dedup rule from code; prompts and the harness import nothing from server**
- Size and keys: ~950 LOC. Keys `fk-10c8c43d9e89`, `fk-9850339b610c`; the upward-import half of
  `fk-20cb93d17ff3`. Priority **high**.
- Files: `fused-memory/src/fused_memory/reconciliation/stage_capabilities.py` (new), `fused-memory/src/fused_memory/reconciliation/prompts/__init__.py`, `fused-memory/src/fused_memory/reconciliation/prompts/stage1.py`, `fused-memory/src/fused_memory/reconciliation/prompts/stage2.py`, `fused-memory/src/fused_memory/reconciliation/prompts/stage3.py`, `fused-memory/src/fused_memory/recon_state/recon_self_model.py` (moved there by S19 β1), `fused-memory/src/fused_memory/reconciliation/stages/base.py`, `fused-memory/src/fused_memory/reconciliation/harness.py`, `fused-memory/src/fused_memory/server/main.py`, `fused-memory/src/fused_memory/memory_metadata.py`, `fused-memory/src/fused_memory/server/grouped_read.py`, `fused-memory/tests/test_recon_self_model.py`, `fused-memory/tests/test_recon_report_guidance_drift.py`, `fused-memory/tests/test_finding_provenance_prompt_guidance.py`, `fused-memory/tests/reconciliation/test_stage_capabilities.py` (new)
- On the size: 15 files; the three outside `reconciliation/` each take a one-site edit (the guidance hand-off
  in `run_server`, `PARENT_ID_KEY` in `memory_metadata.py`, the re-bind in `grouped_read.py`).
- Deps: δ, 9280 (`server/main.py`). (S19 β1, for `recon_state/recon_self_model.py`, is reached through δ.)
- Work: C7 except the move. `capabilities_for(stage)` reads the denial tables, which stay in
  `cli_stage_runner.py`; `stage_capabilities.py` imports them. The stage prompts take `StageCapabilities`; the
  subtask sentence is rendered from the table; the three dedup-anchor passages are replaced by
  `render_dedup_anchor_rule()`; a drift test checks each `MCP_CALL_SIGNATURES` entry's parameter names against
  the recon-report and fused-memory tool registries. `server/main.py::run_server` renders the recon-report tool
  guidance with `render_recon_report_tool_guidance(get_recon_report_tool_signatures())` and passes the string to
  the `ReconciliationHarness` constructor, which hands it to stages through
  `StageCollaborators.report_tool_guidance`. `prompts/stage1.py` imports `CHILD_KINDS` and `PARENT_ID_KEY` from
  `fused_memory/memory_metadata.py` (which gains `PARENT_ID_KEY`), and `server/grouped_read.py` re-binds both
  from it.
- Deletes: the 9 capability literals; the `DISALLOW_SUBTASK_CREATE` sentence; the `_derive_affected_ids` /
  `compute_content_fingerprint` narrations; the function-local `from fused_memory.server.recon_report import`
  in `prompts/__init__.py`; `prompts/stage1.py`'s `from fused_memory.server.grouped_read import`;
  `server/grouped_read.py`'s second `CHILD_KINDS` declaration (SPOT); the "pending ξ cross-check" caveat in
  `recon_self_model.py`.
- Signal: boundary row 16; the live Stage-2 system prompt (rendered by `build_stage2_system_prompt` under the
  service's config) states the dedup anchor in the self-model's words, and its subtask sentence names a real
  table.
- Consumer: every stage prompt; S19's DAG test; S19 ε.
- G7: INV-1 holds (capabilities from the table, not prose); INV-10 `guards-exercise-behaviour` holds (the drift
  test reads the live registries, not prose).

**ι2 — Prompt fragment library moves out of the package __init__ into prompts/fragments.py**
- Size and keys: ~500 LOC of import edits around a 1,390-line move. The library half of `fk-20cb93d17ff3`.
  Priority medium.
- Files: `fused-memory/src/fused_memory/reconciliation/prompts/__init__.py`, `fused-memory/src/fused_memory/reconciliation/prompts/fragments.py` (new), `fused-memory/src/fused_memory/reconciliation/prompts/stage1.py`, `fused-memory/src/fused_memory/reconciliation/prompts/stage2.py`, `fused-memory/src/fused_memory/reconciliation/prompts/stage3.py`, `fused-memory/src/fused_memory/reconciliation/stages/memory_consolidator.py`, `fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py`, `fused-memory/tests/_recon_prompt_write_scope.py`, `fused-memory/tests/server/test_add_memory_write_triage_gate.py`, `fused-memory/tests/server/test_recon_report_citation_repair.py`, `fused-memory/tests/test_dangling_citation_handoff_prompts.py`, `fused-memory/tests/test_duplicate_finding_salvage_guidance.py`, `fused-memory/tests/test_finding_provenance_prompt_guidance.py`, `fused-memory/tests/test_finding_severity_vocabulary.py`, `fused-memory/tests/test_freeze_write_triage_population.py`, `fused-memory/tests/test_recon_amend_tool_advertisement.py`, `fused-memory/tests/test_recon_gate_closure_guidance.py`, `fused-memory/tests/test_recon_leak_cleanup_markup_guard.py`, `fused-memory/tests/test_recon_marker_kind_prompt_guidance.py`, `fused-memory/tests/test_recon_report_guidance_drift.py`, `fused-memory/tests/test_recon_report_stage_gated_tool_advertisement.py`, `fused-memory/tests/test_stage1_consolidation_guidance.py`, `fused-memory/tests/test_stages.py`, `fused-memory/scripts/freeze_write_triage_population.py`
- On the size: 24 files; every edit outside the move is one import line (21 importing modules measured at
  `1b106418da`, one more than at `1f240e6da2`: task 9317's `fused-memory/tests/test_recon_leak_cleanup_markup_guard.py`
  reads `prompts.LEAKED_FRAGMENT_CLEANUP_REASON` off the package; plus
  `fused-memory/scripts/freeze_write_triage_population.py`, which imports `FLAG_FOR_STAGE2_MARKER_KIND` and
  `STAGE2_SUPPRESS_GUARD_KIND` from the package; both would break under "no re-export shim").
- Deps: ι1, κ2 (`task_knowledge_sync.py`, `memory_consolidator.py`).
- Work: heuristic 13 and 14 for `prompts/`: the library is a sibling leaf the stage modules import downward;
  `__init__.py` keeps a docstring and no names; no re-export shim.
- Deletes: the reach-back imports (11 + 12 + 5 names); the library body in `__init__.py`.
- Signal: `python -c 'import fused_memory.reconciliation.prompts as p; print([n for n in vars(p) if not n.startswith("__")])'` prints `[]` (the submodules are not imported by the package); the stage prompts render byte-identical text (a snapshot test).
- Consumer: stage modules, tests, `scripts/legibility/detectors.py` (string mention only).
- G7: INV-9 holds.

**ε — Consecutive-cycle recon alarms as RunStreak declarations over shared.streak.KeyedStreaks**
- Size and keys: ~700 LOC. Key `fk-90e488a174b7`. Priority medium.
- Files: `fused-memory/src/fused_memory/reconciliation/run_streaks.py` (new), `fused-memory/src/fused_memory/reconciliation/harness.py`, `fused-memory/src/fused_memory/reconciliation/task_count_snapshot_cadence.py`, `fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py`, `fused-memory/src/fused_memory/reconciliation/standing_decision_growth_sweep.py` (created by S13 ι), `fused-memory/src/fused_memory/reconciliation/recon_escalations.py` (created by S1 ι1), `fused-memory/tests/reconciliation/test_run_streaks.py` (new), `fused-memory/tests/reconciliation/test_task_count_snapshot_cadence.py`, `fused-memory/tests/reconciliation/test_standing_decision_growth_sweep.py`, `fused-memory/tests/test_harness.py`
- Deps: ι1 (`harness.py`), γ2 (`task_knowledge_sync.py`); S1 ι1 (the recon escalation registry, C8, and
  `FilingPolicy`, C1; lead ruling OQ-7); S13 ι (the growth sweep in `standing_decision_growth_sweep.py`; lead ruling OQ-8). Sibling
  task ids wired at decompose.
- Work: C5, D8. Two declarations, each naming a `ReconEscalation` member ε adds to S1's registry
  (`folds=True`; today's categories, `recon_stale_task_count_snapshot` and the growth-sweep category, and
  today's severities); evaluation in the driver's tail for `FULL`/`BACKLOG_CHUNK`; each firing files through
  `policy_for(member)`, so one filing path remains and `_escalate` gains no parameter; the growth alarm's
  evaluation moves from inside Stage 2 (`TaskKnowledgeSync.run`'s call to `_maybe_escalate_growth_sweep_failures`)
  to the driver's tail. The stage keeps recording its stat. The growth member files through S1's
  `FilingPolicy.open_l1`, at level 1 as today's `Escalation(level=1, severity='blocking')` does (lead ruling
  OQ-7; `_escalate` alone would file at the default level 0,
  `escalation/src/escalation/models.py::Escalation.level`, so the member names the policy).
- Deletes: `extract_snapshot_written`, `compute_snapshot_miss_streak`, `evaluate_snapshot_cadence`,
  `_maybe_escalate_stale_task_count_snapshot`, the growth quartet (in `standing_decision_growth_sweep.py`
  after the split), the stage-side escalation call and its synthetic `recon-esd-growth-sweep:<project>` key and
  `has_open_l1` dedupe, and the comment naming the third streak (it becomes "add a `RunStreak`").
- Signal: boundary row 12; after deploy, the stale-snapshot and growth-sweep escalations on the 8103 queue keep
  their categories and fold repeats (read with `get_pending_escalations` on that queue: one pending record per
  category and project while the condition lasts, its dedupe count rising).
- Consumer: the driver's tail; the recon escalation watcher.
- G7: INV-4 `storm-escape-required` holds through the filer's fold (open and recently-resolved dedupe), not a
  latch: a resolved-but-persisting condition files again on the first bad cycle after the 24-hour
  recently-resolved window (`harness.py::_RESOLVED_RECURRENCE_WINDOW_SECONDS`), instead of never until a good
  cycle (sitting ruling OQ-1); INV-5 holds.

**ζ — Cycle-summary write state as a typed StageReport field**
- Size and keys: ~700 LOC. Key `fk-f23c6e137ce5`. Priority medium.
- Files: `fused-memory/src/fused_memory/models/reconciliation.py`, `fused-memory/src/fused_memory/reconciliation/harness.py`, `fused-memory/src/fused_memory/recon_state/journal.py` (moved there by S19 α), `fused-memory/src/fused_memory/reconciliation/stages/memory_consolidator.py`, `fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py`, `fused-memory/src/fused_memory/reconciliation/summary_pool.py`, `fused-memory/tests/test_harness.py`, `fused-memory/tests/test_stages.py`, `fused-memory/tests/reconciliation/test_stage1.py`, `fused-memory/tests/reconciliation/test_episode_freshness_cursor.py`
- Deps: ε, 7877. (S19 α is reached through α.)
- Work: C4's `CycleSummaryWrite`; writers set it (the Stage-2 writer stays in `TaskKnowledgeSync`, which the
  S13 split leaves in `task_knowledge_sync.py`; `IntegrityCheck` writes no cycle summary); `_cycle_summary_ledger_write_missing` reads it (fires on
  `WRITE_FAILED` and `RECOVERY_UNCONFIRMED`); the journal maps legacy stats (C4 invariant).
- Deletes: the three stat names in code and their `== 0` / `is True` parsing.
- Signal: after deploy, the newest full run's Stage-1 and Stage-2 reports carry `cycle_summary: "written"`
  (journal read); boundary rows 2 and 3 assert the backstop arms.
- Consumer: the backstop (η1 moves it).
- G7: INV-2 holds.

**κ1 — Targeted reconciler: one gated write path, classified outcomes, an event per fail-open arm; deferred writes replayed with causation**
- Size and keys: ~1,300 LOC. Keys `fk-05443fb78da6`, `fk-be8ed2cd9ce9`, `fk-2bb6c913d747`,
  `fk-147a9951edc7`. Priority medium.
- Files: `fused-memory/src/fused_memory/reconciliation/targeted.py`, `fused-memory/src/fused_memory/reconciliation/cancel_backstop.py` (created by task 8144), `fused-memory/src/fused_memory/middleware/task_interceptor.py`, `fused-memory/src/fused_memory/server/main.py`, `fused-memory/src/fused_memory/reconciliation/harness.py`, `fused-memory/src/fused_memory/recon_state/event_buffer.py` (moved there by S19 α), `fused-memory/src/fused_memory/recon_state/journal.py` (moved there by S19 α), `fused-memory/tests/_interceptor_helpers.py` (created by S10 μ1 9278), `fused-memory/tests/test_targeted.py`, `fused-memory/tests/test_task_interceptor.py`, `fused-memory/tests/test_event_buffer.py`, `fused-memory/tests/test_harness.py`, `fused-memory/tests/reconciliation/test_recon_task_write_gate.py` (new)
- On the size: 13 files; `_interceptor_helpers.py` (49 test files construct `TaskInterceptor` through it after
  S10 μ1) takes the new provider member once, and `journal.py` gains one aggregate.
- Deps: ζ (`harness.py`), 9280, 9282, 8144. Not 9285 (§Pre-conditions: it waits on the merge-lane chain; file
  locks order `targeted.py`). S19 α is reached through α.
- Work: C6 for the targeted reconciler (lead ruling OQ-2): the reconciler factory is a provider-callable member
  of S10 μ2's `InterceptorCollaborators`, resolved at the end of `TaskInterceptor` construction;
  `_gated_write` over `write_failure_reason`; a refusal writes `run_actions` `write_refused`; each of the 6
  log-only arms writes `run_actions` `fail_open:<arm>`, and `ReconciliationJournal.get_stats` reports an
  aggregate `fail_open` count per arm over its window, so an arm firing 100 times an hour is one number on the
  stats surface rather than 100 unread rows (INV-4); a failed replay calls a new
  `EventBuffer.release_deferred_write(id)` that increments `attempt_count` and unclaims it, so the next cycle
  retries; an exhausted row files `recon_integrity_issue` before deletion; replay passes
  `metadata.pop('_causation_id')` as `causation_id` (`deferred_writes` has no causation column; the id is
  stored in `metadata['_causation_id']` by `targeted.py::TargetedReconciler._fenced_add_memory`). The guard
  test covers `targeted.py` and `cancel_backstop.py`.
- Deletes: the `task_interceptor = None` attribute and both post-construction assignments in `run_server`;
  `_on_task_blocked`'s raw `taskmaster.update_task` fallback; the two "not wired; skipping" guards (one of
  them in `cancel_backstop.py` after task 8144); the per-site classifications at the 7
  `interceptor_write_succeeded` call sites (13 mentions).
- Signal: boundary rows 13–14; after deploy, `run_actions` rows of type `fail_open:*` and `write_refused`
  appear for targeted runs on the live journal when their arms fire; a deferred write that failed once is
  replayed on the next cycle (journal + `deferred_writes` read).
- Consumer: κ2 (the guard), the L2 watcher (the exhausted-write escalation).
- G7: INV-11 holds (every fail-open arm leaves a record); INV-4 not met: the six fail-open arms gain a
  `run_actions` row and an aggregate `get_stats` count, not a threshold escalation; INV-7
  `holds-owned-and-bounded` holds (a claimed deferred write is released and its retries bounded).

**κ2 — Reconciliation stages and harness write tasks only through the TaskInterceptor**
- Size and keys: ~500 LOC. Key `fk-875cca636438`. Priority medium.
- Files: `fused-memory/src/fused_memory/reconciliation/harness.py`, `fused-memory/src/fused_memory/reconciliation/stages/base.py`, `fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py`, `fused-memory/src/fused_memory/reconciliation/policies/briefing_refresh.py` (created by S13 ι), `fused-memory/src/fused_memory/server/main.py`, `docs/legibility/exception-ratifications.yaml`, `fused-memory/tests/test_fused_memory_governed_lists.py`, `fused-memory/tests/test_stages.py`, `fused-memory/tests/reconciliation/test_task_knowledge_sync_citation_verify.py`, `fused-memory/tests/test_task_text_markup_internal_writers.py`, `fused-memory/tests/reconciliation/test_recon_task_write_gate.py`
- Deps: κ1; S13 ι (the briefing hook in `policies/briefing_refresh.py`; S13 task id wired at decompose;
  lead ruling OQ-8).
- Work: C6 for the harness and stages (D10; lead ruling OQ-4); `StageCollaborators.task_writes`; the guard extends to
  all of `fm/reconciliation/` with one exemption, `consolidation_auto_executor.py` (lead ruling OQ-9): the exemption list
  is declared as an INV-12 governed list in `fused-memory/tests/test_fused_memory_governed_lists.py` with
  `default=Policy('inv12-recon-hazard-gate-raw-backend')`, and that row is added to
  `docs/legibility/exception-ratifications.yaml` (source: `plans/memory-auto-consolidation-prd.md` D8 and the
  lead's ruling S12/OQ-9; covers: reconciliation's hazard-gate filer writes tasks on the raw backend by
  design).
- Deletes: `queue_briefing_refresh_tasks`' raw `add_task` call and its local `DuplicateCandidateKeyError` fold
  (the curator dedups).
- Signal: boundary row 15; `scripts/exception_register.py` reports the guard's exemption as `Policy`; after
  deploy, a "Refresh briefing: …" task, when one is queued, has a curator ticket and a `task_created` event
  (ticket store and event buffer reads).
- Consumer: the curator, reconciliation's event stream.
- G7: INV-1 and heuristic 10 *uniformly* hold (one mechanism, a guard test); INV-12
  `exceptions-owned-or-ratified` holds: the guard is an allow-list whose one entry carries a `Policy`
  disposition naming a ratification row.

**η1 — harness.py partition I: the cycle-summary backstop and run recovery in their own modules**
- Size and keys: ~1,200 LOC changed around a ~1,100-line move. Contributes to `fk-a88506a1e327` and
  `fk-16baf39b1db6`. Priority medium.
- Files: `fused-memory/src/fused_memory/reconciliation/cycle_summary_backstop.py` (new), `fused-memory/src/fused_memory/reconciliation/run_recovery.py` (new), `fused-memory/src/fused_memory/reconciliation/harness.py`, `fused-memory/tests/reconciliation/test_cycle_summary_backstop.py` (new), `fused-memory/tests/reconciliation/test_run_recovery.py` (new), `fused-memory/tests/test_harness.py`, `fused-memory/tests/test_halt_visibility.py`
- Deps: κ2.
- Work: D9 (backstop, recovery), D11. The 29 dotted patches of `write_stage{1,2}_cycle_summary` into the harness
  module become a constructor-injected writer on the backstop.
- Deletes: those symbols from `harness.py`; the moved tests from `test_harness.py`.
- Signal: `wc -l` of `harness.py` falls by the moved span; `python -c 'import fused_memory.reconciliation.run_recovery'` imports no `harness`; startup recovery logs unchanged on the next restart.
- Consumer: harness.
- G7: heuristic 13 per new file (no import of `harness`).

**η2 — harness.py partition II: cycle diagnostics, the finding gate and BacklogIterator in their own modules; prose rehomed**
- Size and keys: ~1,400 LOC changed around a ~1,450-line move. Key `fk-a88506a1e327` (closes);
  contributes to `fk-16baf39b1db6`. Priority medium.
- Files: `fused-memory/src/fused_memory/reconciliation/cycle_diagnostics.py` (new), `fused-memory/src/fused_memory/reconciliation/finding_gate.py` (new), `fused-memory/src/fused_memory/reconciliation/backlog_iterator.py` (new), `fused-memory/src/fused_memory/reconciliation/harness.py`, `fused-memory/tests/reconciliation/test_cycle_diagnostics.py` (new), `fused-memory/tests/reconciliation/test_finding_gate.py` (new), `fused-memory/tests/reconciliation/test_backlog_iterator.py` (new), `fused-memory/tests/test_harness.py`, `fused-memory/tests/reconciliation/test_throughput.py`, `fused-memory/tests/reconciliation/test_finding_task_escalation.py`
- Deps: η1.
- Work: D9 (the rest), prose rehome under the Comments stance; `BacklogIterator`'s constructor gains the recon
  filer; `harness.py::ReconciliationHarness._project_loop` constructs it from its new module (a downward
  import; `fused-memory/tests/test_harness.py`, which imports it at module level, repoints). The stays-whole
  measurement of the remainder goes in the commit message.
- Deletes: the moved symbols; the reviewer-answering prose blocks named in D9.
- Signal: `harness.py` has no `class BacklogIterator`; the live backlog path logs `Backlog chunk N` lines
  from the new module on the next backlog.
- Consumer: harness (`_project_loop`).
- G7: heuristic 13 per new file.

**θ — Reconciliation harness tests drive public seams; the remaining private test seams are deleted**
- Size and keys: ~1,200 LOC. Key `fk-16baf39b1db6` (closes). Priority medium.
- Files: `fused-memory/src/fused_memory/reconciliation/harness.py`, `fused-memory/tests/test_harness.py`, `fused-memory/tests/test_stages.py`, `fused-memory/tests/reconciliation/test_stage1.py`, `fused-memory/tests/reconciliation/test_finding_task_escalation.py`, `fused-memory/tests/reconciliation/test_preservation_specimen_guard.py`, `fused-memory/tests/reconciliation/test_task_count_snapshot_cadence.py`, `fused-memory/tests/test_standing_decision_storm_escape.py`, `fused-memory/tests/test_recon_gate_closure_guidance.py`, `fused-memory/tests/test_halt_visibility.py`, `fused-memory/tests/reconciliation/recording_escalation_queue.py` (new)
- Deps: η2. Not ι2: the two share only test files (`fused-memory/tests/test_stages.py`,
  `fused-memory/tests/test_recon_gate_closure_guidance.py`), rebase-only under the lead's test-file rule.
- Work: the harness takes an injected `sleep`; tests read filed escalations from a recording queue passed as
  the `escalation_queue` constructor argument (task 6463's seam) instead of mocking `_escalate`; the 16
  `harness._known_projects[…] = …` writes in `test_harness.py` (measured at `1b106418da`) pass the registry
  through the existing `known_projects=` constructor argument and read it through the read-only
  `known_projects` property task 6158 added (`36e4f5e860`); tree fetches go through the injected task port.
  Re-run the measurement rule.
- Deletes: `stages: list[Any]`; module-level `_sleep`; the test patches of `_escalate`, `_make_stages`,
  `_sleep` and `stages`; the test writes into `_known_projects`.
- Signal: the measurement rule re-run reports no access to `_escalate`, `_make_stages`, `_run_remediation_pass`,
  `stages`, `_sleep` and no dotted patch into the harness module (θ's commit records the residual count).
- Consumer: every future harness test.
- G7: tests stance; INV-10 holds.

**ω — Integration gate: the run-lifecycle boundary suite and the live journal check**
- Size: ~600 LOC. Keys: none (gate). Priority medium. Signal leaf.
- Files: `fused-memory/tests/reconciliation/test_run_lifecycle_boundary.py` (new), `fused-memory/scripts/recon_run_lifecycle_check.py` (new)
- Deps: θ, ι2.
- Work: boundary rows 1–17 and 10a–10d as one suite through `drive`, the stage builders, a scripted stage runner and the
  server's identity path; a read-only script over `data/reconciliation/reconciliation.db` and the write journal
  that reports, for the last N runs, the D2/D4/C3/C4 properties (D4 includes `events_processed`; C3 includes
  every run-tied `write_ops` row and creation-journal row carrying a `recon-stage-*` identity) and exits
  non-zero on a violation.
- Signal: the suite passes in CI; `uv run --project fused-memory python fused-memory/scripts/recon_run_lifecycle_check.py --runs 20` exits 0 on the live store after deploy.
- Consumer: the operator; the next `/review-all`.
- G7: INV-13 `readers-prove-their-producer` holds: the script reports "no run written since <deploy sha>" as
  its own state, distinct from "all properties hold", when the journal has no run newer than the deploy.

## Rationale rehomed (the Comments stance)

Leaves that delete or move prose (β2, γ1, γ2, η1, η2) move reviewer-answering prose and dated measurements to
their commit messages, leaving at most a one-line pointer to this PRD's decision (D2, D4, D9). A task number in
a comment survives only where it names live provenance a reader needs.

## Out of scope

- The `task_knowledge_sync.py` partition, task 6846's `IntegrityCheck` extraction, flag identity, the edge-sweep
  primitive and `ReconReportState` (S13).
- The fused-memory MCP server's `ToolContext`, the dispatch wrapper and the `TaskInterceptor` builder (S15); δ
  edits two helper functions at the home S15 α gives them (lead ruling OQ-11).
- `escalation/filing.py`, `FilingPolicy` and the recon escalation registry (S1); ε adds one registry member
  per streak alarm and chooses a policy only (lead ruling OQ-7).
- Moving modules into `fused_memory.recon_state` (S19).
- Reading the run binding on the recon-report MCP server, refusing a differing `run_id` there, and dropping
  the prompt's `run_id` for recon-report calls (S13 ξ, which depends on δ for the headers; lead ruling OQ-10
  and the OQ-3 ruling's cross-stream rule).
- Any change to how a non-reconciliation client passes `agent_id` (task agents, interactive sessions,
  scripts, skills), or to any prompt or skill other than the sweep files δ names (the OQ-3 ruling).
- Re-routing the auto-consolidation hazard-gate filer through the curator (`plans/memory-auto-consolidation-prd.md`
  D8 stands; lead ruling OQ-9).
- Changing what any stage does with its inputs, the streak thresholds, or any escalation category.

## Open questions (tactical)

1. **Where `RunError.sqlite` comes from.** Keep `sqlite_classification(exc)`'s keys verbatim, or a typed
   sub-record. **Suggested:** verbatim mapping. Decide in α.
2. **`StageInputs` field names.** Keep today's attribute names as field names to keep test diffs small.
   **Suggested:** yes. Decide in γ2.
3. **`run_actions` vocabulary for κ1.** `write_refused` / `fail_open:<arm>` as operation strings, or a typed
   action kind. **Suggested:** strings matching the existing `run_actions.operation` column. Decide in κ1.
4. **The check script's output format.** Text table and a JSON mode. **Suggested:** both. Decide in ω.
5. **How C3's refusal reaches the caller.** The resolver raises an exception carrying
   `StageIdentityRefusal`, serialized as task 4867's unknown-argument refusal is
   (`shared/src/shared/mcp_unknown_arguments.py::unknown_argument_refusal`, JSON in the error text), or each
   resolver call site returns `as_error()`. Either way the caller reads the same fields (boundary row 10a)
   and the predicate stays one function. **Suggested:** the exception, so no call site carries the rule.
   Decide in δ.

Every design-level question was ruled at the 2026-11-12 sitting (§Rulings); none is open.
