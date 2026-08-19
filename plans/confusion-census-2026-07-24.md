# confusion census 2026-07-24

Project: dark_factory

## Saturation

- batches: 18
- stop reason: saturated
  - batch 0: dup_rate=0.75 (total=20, succeeded=20, failed=0, saturated=False)
  - batch 1: dup_rate=0.95 (total=20, succeeded=20, failed=0, saturated=True)
  - batch 2: dup_rate=0.84 (total=20, succeeded=19, failed=1, saturated=False)
  - batch 3: dup_rate=0.79 (total=20, succeeded=19, failed=1, saturated=False)
  - batch 4: dup_rate=0.80 (total=20, succeeded=20, failed=0, saturated=False)
  - batch 5: dup_rate=0.89 (total=20, succeeded=19, failed=1, saturated=False)
  - batch 6: dup_rate=0.89 (total=20, succeeded=19, failed=1, saturated=False)
  - batch 7: dup_rate=0.90 (total=20, succeeded=20, failed=0, saturated=True)
  - batch 8: dup_rate=0.84 (total=20, succeeded=19, failed=1, saturated=False)
  - batch 9: dup_rate=0.65 (total=20, succeeded=20, failed=0, saturated=False)
  - batch 10: dup_rate=0.70 (total=20, succeeded=20, failed=0, saturated=False)
  - batch 11: dup_rate=0.79 (total=20, succeeded=19, failed=1, saturated=False)
  - batch 12: dup_rate=0.74 (total=20, succeeded=19, failed=1, saturated=False)
  - batch 13: dup_rate=0.80 (total=20, succeeded=20, failed=0, saturated=False)
  - batch 14: dup_rate=0.65 (total=20, succeeded=20, failed=0, saturated=False)
  - batch 15: dup_rate=0.80 (total=20, succeeded=20, failed=0, saturated=False)
  - batch 16: dup_rate=0.95 (total=20, succeeded=20, failed=0, saturated=True)
  - batch 17: dup_rate=0.94 (total=20, succeeded=18, failed=2, saturated=True)

## Origin x Manifestation Matrix

| origin \ manifested | architect | implement | verify | review | recon | ops | unknown |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prd | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| decompose | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| architect | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| implement | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| review | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| recon | 0 | 0 | 0 | 0 | 12 | 0 | 0 |
| ops | 0 | 0 | 1 | 0 | 0 | 5 | 1 |
| unknown | 4 | 2 | 0 | 0 | 4 | 1 | 14 |

## Synthesis

**Census synthesis is complete, but I could not save the report file** — this unattended session has no write permission for the repo (both the Write tool and shell writes to `plans/` hit unanswered approval gates, which is itself an instance of census cluster 1.6 below). The full report is delivered here for saving to `plans/confusion-census-2026-07-24.md`; I've left a memory-trail note so a future session picks this up. Still outstanding after the file is saved: the docs-only commit, curator-path filing of remediation candidates R1–R14, codebook merger dispositions (§5), and the `census-state.json` advance.

**TLDR of the census itself:** 52 verified findings across 45 sessions form 9 clusters + 9 one-offs. The three headline masses: (1) the confusion pipeline's digest generator captures the trickle-coder's own prompts as session content — 12 sessions, the largest cluster, and self-amplifying since contaminated digests get re-mined; (2) reconciliation runs are unauditable from their own reports (`target: "?"` journal rows, `stats` vs `_reported` contradictions, `tokens_used: 0`) — 9 sessions; (3) session-start memory injection returns unusable context in both directions (anaphoric contentless facts, high-scored irrelevant hits) — 5 sessions. Zero merge-manifested sightings this cycle, so the PRD's architect/implement→merge hypothesis remains untested.

The report, verbatim:

---

# Confusion census — 2026-07-24

**Date:** 2026-07-24
**Method:** periodic census per `plans/confusion-reduction-prd.md` §5 (η): stratified-random saturation mining (Sonnet) over session digests → per-finding verification against current main (Sonnet) → this synthesis (Fable). Every finding restated below survived the verification stage; this report adds clustering and counting only — no diagnosis appears here that was not itself verified.
**Companion artifact:** `docs/legibility/confusion-codebook.yaml` (dispositions in §5 are inputs to the merger, which promotes/rejects in place).
**Run notes:** first completed periodic census. The 2026-07-22 attempt aborted in the verify/synthesis stages (120s CLI timeout; remediation tasks 2951–2953). The previous baseline is the 2026-07-13 big-bang survey (`plans/agent-legibility-survey-2026-07-13.md`).

## Corpus

- **52 verified findings across 45 unique sessions** (7 sessions contributed two findings each).
- Composition note, visible directly from the sightings: the mined sample over-represents the legibility pipeline's own sessions. 19 of 52 findings (37%) are about the confusion-reduction pipeline itself (trickle-coder digests, census verifiers, harness friction inside those sessions), and 14 of 52 (27%) are about reconciliation-stage runs. Orchestrated task-pipeline phases (architect/implement/review) account for most of the remainder; **zero findings manifested in the merge phase this cycle**.
- The PRD's motivating hypothesis — that merge/verify-manifested confusion originates in architect/implement phases — is **untested by this batch**: only one finding manifested in verify and none in merge. The matrix reports this rather than guessing.

## Executive summary (observations)

1. **The confusion pipeline's largest observed confusion source this cycle is itself.** The single biggest cluster (12 sightings across 12 distinct sessions) is the digest generator capturing the trickle-coder's own task prompt as session content labeled "User Corrections", recursively nesting prior digests inside new ones. Three further sightings show digest bodies decoupled from their scores (high-scored sessions rendered as boilerplate or empty), and three show the census verifier lacking an operating envelope (sandbox scope, per-command approvals, unknown transcript paths). Part of this concentration is sampling — the pipeline mines its own sessions — but the defects are real and self-amplifying: contaminated digests are re-mined by later runs.
2. **Reconciliation runs cannot currently be audited from their own outputs.** Ten sightings across nine sessions show three verified facets: `write_journal` rows with `target: "?"` plus duplicate coarse/fine entries; `stats` vs `_reported` counter blocks that disagree (including a top-level counter contradicting its own nested stats in the same report); and `tokens_used: 0` alongside nonzero `llm_calls` in every stage of two runs.
3. **Session-start memory injection is returning unusable context in both observed failure directions**: Graphiti facts that are anaphoric/self-referential ("the decision was made to fix via citations, choosing Option A over Option B" — never naming the subject), and retrieval hits scored 0.75–1.0 relevance that are unrelated to the task's domain. Five sessions, spanning architect and implement manifestations.
4. **Two verified within-cycle repeats of previously known shapes:** the `watcher-rearm.sh` `DARK_FACTORY_ROOT` env-var contract gap recurred in three separate rotations, and the recon terminal-task write guard's single-field corrective path was verified as a second confirmed occurrence (first seen 2026-07-10/11 on task 544, now on autopilot_video tasks 644/650) — the finding itself documents that each recurrence had been re-diagnosed as a fresh bug.
5. **Phase-stamp coverage is weak in this batch**: 25/52 findings carry `origin_phase: unknown` and 15/52 `manifested_phase: unknown`, concentrated in the legibility pipeline's own sessions (which run outside the orchestrator's phase-stamped lifecycle). Improving stamping for non-task sessions would make the next matrix more informative.

## Origin × manifestation matrix

Rows = `origin_phase`, columns = `manifested_phase`. Counts are verified sightings (52 total). `merge` is kept explicitly to show its zero.

| origin \ manifested | architect | implement | verify | review | merge | recon | ops | unknown | **total** |
|---|---|---|---|---|---|---|---|---|---|
| prd | · | · | · | 1 | · | · | · | · | **1** |
| decompose | 1 | · | · | · | · | · | · | · | **1** |
| architect | 2 | · | · | · | · | · | · | · | **2** |
| implement | · | 1 | · | · | · | · | 1 | · | **2** |
| review | · | · | · | 2 | · | · | · | · | **2** |
| recon | · | · | · | · | · | 12 | · | · | **12** |
| ops | · | · | 1 | · | · | · | 5 | 1 | **7** |
| unknown | 4 | 2 | · | · | · | 4 | 1 | 14 | **25** |
| **total** | **7** | **3** | **1** | **3** | **0** | **16** | **7** | **15** | **52** |

Readings (observational):

- The diagonal dominates: recon-originated confusion manifests in recon (12/12), ops in ops (5/7), review in review (2/2). This batch shows almost no cross-phase propagation — with two verified exceptions: a PRD-authoring defect (stale BLUF) surfacing at review time, and a decompose-time coordination directive being overridden at architect time.
- The `unknown→unknown` mass (14) is almost entirely the legibility pipeline's own sessions, which have no phase-stamped lifecycle. The `unknown→architect` column (4) is session-start memory injection plus worktree provisioning — origin genuinely not localizable from the sightings.

## 1. Verified clusters

Ordered by sighting count. Session ids abbreviated to 8 chars.

### 1.1 Digest pipeline self-ingestion — trickle-coder prompts captured as session content (12 sightings, 12 sessions)

Sessions: 3275ca01, 3b1cd07f, c5522685, ee633c8a, 8d1a9af8, 617d100e, 3f6c6536, dc6036ed, 0abb0308, 8db2d245, 554e62f1, a3d49c3c.

The session-digest generator that feeds the trickle-coder (`plans/confusion-reduction-prd.md` §7.3) captures the trickle-coder's own harness-injected task prompt — instructions, codebook index, and the nested prior session digest it contains — verbatim under a "## User Corrections" heading. Twelve independent sessions show the same shape. Verified consequences observed in the sightings:

- The corrections signal is polluted with recursive, zero-signal content: several digests' *sole* "User Correction" is the meta-prompt, with all `signal_counts` at zero (617d100e, 0abb0308) — i.e., the digest's own signals confirm no correction occurred.
- Nesting compounds: one digest (554e62f1) embeds a second, unrelated session's full digest (1d0d5d85, a recon integrity-check report) inside the mislabeled turn, making the session it nominally describes unassessable.
- One sighting (a3d49c3c) shows the same mislabeling for an ordinary architect task-assignment block ("# Context / ## Agent Identity / # Task"), indicating the heuristic buckets early user-role turns as corrections generally, not just trickle-coder prompts.
- Because the mining corpus does not exclude prior trickle-coder/meta sessions, the pipeline re-mines its own scaffolding: this census batch itself contains twelve such sightings.

Evidence (representative): `## User Corrections — (turn 3) You are the trickle coder for the dark-factory agent-confusion codebook (plans/confusion-reduction-prd.md §7.3). Read the session digest below and decide which existing codebook entries it matches...`

### 1.2 Reconciliation runs are not auditable from their own reports (10 sightings, 9 sessions)

Sessions: 47959768, 3802d188, a8dbb27b, 6ff1311a, 33b115bf, 459a4ae3 (×2), dbfab1c8, 40dff74e, 3bfa5bdc.

Three verified facets, all in the recon stage-report / `write_journal` surface:

**(a) Write-journal provenance holes (4 sightings: 47959768, 3802d188, a8dbb27b, 6ff1311a).** The MCP action journal emits, per logical mutation, a generic row (`operation: "add"/"delete"/"remove_edge"`, `target: "?"`) immediately adjacent to a specific row (`add_memory`/`delete_memory`/`add_system_record`/…) carrying the real UUID. Some operation kinds (`update_edge`, `add_episode` in a8dbb27b) get *only* `"?"` rows. In one run (6ff1311a) 8 of 15 logged actions carried no resolvable target. A reviewer must correlate rows by millisecond timestamp proximity to reconstruct one operation; the generic rows are unattributable on their own.

**(b) `stats` vs `_reported` counter divergence (4 sightings: 33b115bf, 459a4ae3, dbfab1c8, 40dff74e; corroborated in a8dbb27b, 6ff1311a).** Stage reports carry two accounting layers that are never cross-checked before emission. Observed contradictions: `stats.memories_deleted: 0` vs `_reported.memories_deleted: 3.0` (33b115bf); `memories_added: 3` vs `_reported.memories_added: 2.0` (459a4ae3); top-level `items_flagged: 1` vs the same report's `stats.items_flagged: 0.0` (dbfab1c8); `_reported` introducing keys absent from the stats schema (`memories_written`); and in a8dbb27b the journal's 4 logged `add_episode` calls vs `stats.episodes_added: 0`. In the benign case (40dff74e) `_reported` merely duplicates the top-level counter under a different name — still leaving a consumer no signal for which figure is authoritative.

**(c) Cost telemetry unwired (2 sightings: 3bfa5bdc, 459a4ae3).** Every stage in both runs reports real `llm_calls` counts (35/23/39 and 21/11/33) with `tokens_used: 0` — token accounting appears never wired into the stage runner's metrics emission, so per-cycle spend is invisible to run reviewers and to any downstream cost/capacity decision.

Evidence (representative): `{"operation": "delete", "target": "?", "source": "write_journal", ...}` followed 10ms later by `{"operation": "delete_memory", "target": "28a8413b-...", ...}`; `"llm_calls": 21, "tokens_used": 0`.

### 1.3 Session-start memory injection returns unusable context (5 sightings, 5 sessions)

Sessions: 9de9e14f, 60627405, fc5e835d (facet a); c4c00b6a, d8b662df (facet b).

Two distinct verified facets of the same consumer surface (the mandated session-start Project Context / Conventions / Recent Decisions / Task Context assembly):

**(a) Anaphoric, contentless facts.** Graphiti's episode-to-fact condensation drops the antecedent/subject, producing meta-referential text that confirms a decision/precedent exists without stating what it says: *"The decision was made to fix via citations, choosing Option A over Option B"*; *"Task 2584 serves as a precedent for resolving task 2585"*; *"This decision is explicitly called out in docs/plan-scoring-and-judge.md to prevent relitigation"* (the decision is never named). In the fc5e835d sighting all four context buckets were fully populated and inert — the agent must silently discard them and read source, with no signal that the memory layer under-delivered.

**(b) High-scored but task-irrelevant hits.** Returned items carry `relevance_score` 0.75–1.0 yet are unrelated to the task's domain: a task about `InFlightMergeRegistry` duplicate-coalescing received task-1470 audit wiring, task-841 view scoping, and an unrelated AskUserQuestion decision (c4c00b6a); a task wiring a Qdrant offline-lane pytest config received steward terminal-decision conventions and lambda decision-tree facts (d8b662df). The scoring reflects generic recency/salience rather than fit, and the numeric score lends false confidence.

Both facets consume context budget at every session start under CLAUDE.md's "Starting a session" step 1.

### 1.4 Digest evidence quality — scores decoupled from bodies (3 sightings, 3 sessions)

Sessions: 3b1cd07f (nested digest 0a92e754), 0abb0308 (nested digest b2b4156a), 06dac3e5.

Where 1.1 is about *wrong* content, this cluster is about *missing* content behind the numbers the digest asserts:

- A digest scored 8.0 with `tool_error: 1, not_found: 2` whose rendered body is exclusively injected system-reminder boilerplate (deferred-tool list, agent roster, skills catalogue) — zero dialogue or tool-call content, so content selection is evidently anchoring on message position/type rather than the signals it summarizes.
- A digest scored 10.0 that is 294 bytes: frontmatter only, no body, all `signal_counts` zero — the scoring step and the evidence-extraction step disagree about the session with no consistency check between them.
- Signal rendering that keeps only the trigger phrase and drops the matched substring: `Guard Trips: (turn 21) blocked:` / `Not Found: (turn 21) does not exist` — the guard name, path, or symbol that followed in the source line is never captured, so every occurrence collapses to an uninformative template.

In each case the downstream consumer (trickle-coder or census miner) is left unable to substantiate any classification for a session the pipeline's own scoring flags as significant.

### 1.5 `watcher-rearm.sh` requires `DARK_FACTORY_ROOT`, absent from the rotation launch contract (3 sightings, 3 sessions)

Sessions: 2a08d615, f41dd7df, d7d196be.

Escalation-watcher rotations are spawned by the supervisor with `ROTATION_ESCALATIONS`/`ROTATION_HOURS`/project root/queue path injected as prose, but not the `DARK_FACTORY_ROOT` env var that `scripts/watcher-rearm.sh` hard-requires. The supervisor prompt says "Project root: <path>" — a differently-named concept the agent must map to the env var by inference. In all three rotations the agent discovered the requirement only via a first-call exit-2 failure (`watcher-rearm.sh: DARK_FACTORY_ROOT must be set to a valid directory`), then set the var manually and retried. Three independent rotations hitting the identical first-call failure within the window makes this the cycle's clearest fixed-cost recurring tax: every rotation pays one failed invocation before the bounded-wait mechanism can run.

### 1.6 Census-verifier operating envelope — sandbox scope, approval gates, unknown transcript paths (3 sightings, 3 sessions)

Sessions: e41e6bc1, 96ee5257, 06dac3e5.

The census-verification role is dispatched without the operating envelope its task requires:

- Its prompt instructs confirming cross-session claims via targeted reads, but the flagged sessions' transcripts live outside the allowed working directory; five consecutive `find`/`ls` attempts on `~/.claude/projects` were sandbox-blocked before the agent fell back to empirical reproduction (e41e6bc1).
- Read-only forensic commands (`grep -rn` over the repo, `find`, `git log`) each trip an individual interactive approval gate with no human present, stalling verification on exactly the reads it needs (96ee5257).
- Nothing tells the verifier the deterministic transcript path convention (`data/orchestrator/agent-transcripts/<task_id>/<worktree>/<session>.jsonl.gz`); one session issued ~11 escalating filesystem-wide `find`/`grep` attempts — each blocked or requiring approval — before locating the file and running `digest.py` on it directly (06dac3e5).

Adjacent harness observation from the same sessions, recorded under one-offs (§2): the Bash static-analysis pre-check rejecting multi-line reproduction commands, which the verifier had to distinguish from the runtime bug it was reproducing. (Post-verification corroboration: this synthesis session itself was blocked from writing this report to `plans/` by the same unattended permission gates.)

### 1.7 Recon consolidation matches on surface similarity, not verified semantics (3 sightings, 2 sessions)

Sessions: f3fe6351 (×2), 443c02f0.

Three verified instances of one shape — a similarity/keyword surface match standing in where the guard or claim is defined semantically:

- Stage 1's duplicate-detection search operates over a narrower radius than the actual duplicate footprint, then asserts a cluster "fully resolved" after merging only the subset it found; a same-cycle broader spot-check surfaced 4 more live, un-merged duplicates on the identical topic. The finding notes the topic "keeps recurring across weeks despite the exact command being documented" — a repeat failure mode, not a one-off miss.
- The consolidator canonicalizes near-duplicate `procedural_knowledge` records on textual similarity with no empirical re-verification: the canonical record and several live copies assert one behavior for a git command ("reliably exits 1 but still stages the intended files correctly") while another live record on the same topic asserts materially different behavior; nothing in the write/merge path catches the contradiction.
- Stage 2 assigned `category='cross_project_routing'` because a finding's text cited a task in another project — not because it met the Cross-Project Routing Guard's defined trigger (actual `get_task`/`get_tasks`/`get_statuses` project_id mis-stamping).

### 1.8 Review-round adjudication: prior-round suggestion records carry no resolution status (2 sightings, 2 sessions)

Sessions: 459234ec, 7a14b442.

The PRIOR-ROUND suggestions block handed to the settled/not-settled judge in the review/amendment loop (`merge_disposition.py`) lists every concern raised in earlier rounds with no machine-readable field separating "applied / deliberately kept" from "deferred to a follow-up task and still open." The adjudicator must infer resolution status from prose resemblance, so a current suggestion matching a merely-deferred prior concern risks being mis-marked settled and silently suppressed. In both observed rounds the harness compensates with explicit prompt prose ("…others were merely DEFERRED … and are still OPEN … A current suggestion that matches only a DEFERRED, still-open prior concern has NOT been settled and must still be emitted"), i.e. the mitigation currently lives in judge instructions rather than in the record's schema.

### 1.9 Modules past the Read tool's 256KB ceiling force degraded navigation (2 sightings, 2 sessions)

Sessions: 81d54a47 (architect), b8cedfca (implement).

Two files in the orchestrator package have grown past the Read tool's 256KB cap: `merge_queue.py` (739.6KB observed) and `tests/test_merge_queue.py` (1001.7KB observed). Consequences verified in the sightings: architect-stage whole-file exploration fails mid-plan with no upfront size signal, forcing blind offset/limit reading of `consume()` call sites; and in the implement-stage session one structural fact (file size) cascaded into three separate tool-contract failures — Read cap, ripgrep's no-lookaround limitation on the natural disambiguating pattern, and Edit collisions on near-identical match sites.

## 2. One-off sightings (9 findings, verified; no within-cycle repeat)

- **Self-revising analysis doc left contradictory verdicts live** (6eacb171, prd→review). A long-form PRD-substrate analysis was revised incrementally as §9.4/§11.1/§11.3/§11.5 each retracted a load-bearing claim from §5/§6/§7/§9.5 — but the BLUF and early sections (the parts actually consulted for a decision) were never rewritten or forward-pointed, so the document simultaneously asserts the pre-correction and post-correction verdicts.
- **Deferred MCP tool called before ToolSearch discovery** (bdda3cfb, unknown→ops). With the schema withheld, the agent invented parameter names/types for `Monitor` (`onlyTaskIds`, stringified arrays) and failed client-side validation; nothing prompts checking for deferred-tool schemas before first use.
- **Resume-after-escalation template assumes `.task/iterations.jsonl` exists** (9de9e14f, implement→implement). The standard resume prompt instructs reading `.task/plan.json` and `.task/iterations.jsonl`; the latter is not guaranteed for every task/workflow shape, producing an immediate failed tool call at the start of resumed work.
- **Bash static-analysis pre-check conflatable with the runtime bug it gates** (e41e6bc1, unknown→unknown). Multi-line heredoc reproduction commands were rejected pre-execution ("Contains shell syntax that cannot be statically analyzed") rather than run — a distinct failure mode from the runtime argument-flattening being reproduced, and easy to conflate from error text alone.
- **`resolve_ticket` timeout carries no outcome signal** (85f4d0c5, ops→ops). A 115s timeout returned only "The operation timed out" with no indication whether the resolution applied, pushing the agent into 13 `get_pending_escalations` + 3 `get_scheduler_state` corroborating polls to infer the outcome.
- **Rotation-hour budget invisible during blocking re-arm waits** (ad83edac, ops→ops). `ROTATION_HOURS=4.0` is injected once as prose; each `watcher-rearm.sh` call blocks up to 3600s and reports only per-call outcomes, never cumulative elapsed time. The agent re-armed 7 consecutive times (~7h against a 4h budget) until a human interrupted.
- **Recon terminal-task write guard has exactly one corrective path — second confirmed occurrence** (ec1611ea, unknown→recon). `recon_write_policy.py`'s `ReconTerminalWriteRejected` guard authorizes only `set_task_status_done_provenance_repair` (scoped to `done_provenance`); no analogous path exists for other terminal-task string fields such as top-level `details`. First seen 2026-07-10/11 (task 544), recurred 2026-07-22/23 (autopilot_video 644/650); the finding documents that each recurrence was re-diagnosed as a fresh bug because the boundary is undocumented and untracked.
- **Architect explore-phase hit missing deps in a fresh task worktree** (92d5f938, unknown→architect). Exploration commands in `.worktrees/2884` failed with `ModuleNotFoundError` and command-not-found — consistent with dependencies/tooling not confirmed installed before dispatch — costing exploration turns before the session was cut short by user interrupt with no plan confirmed.
- **Architect overrode an explicit in-task coordination directive** (287f6ff2, decompose→architect). Task 2553's description carried an explicit branching instruction (build on dependency 2248's typed StewardOutcome channel if landed); the architect's plan instead patched only the old path while its own rationale cited 2248 as done/merged. A human rejected the plan-tools call to stop the divergence; no automated check caught it.

## 3. Cross-cutting observations

These restate patterns visible across multiple verified clusters; they diagnose nothing beyond what the sightings show.

1. **Self-observation without self-exclusion.** The pipeline that measures confusion is contaminating its own corpus (1.1), losing the evidence behind its own scores (1.4), and dispatching its verifiers without the access their task requires (1.6). None of these were visible before the pipeline existed; all are now the largest single source of sightings. Any corpus-level trend analysis across cycles should exclude or segment legibility-pipeline sessions until 1.1 is fixed, or the next census will partly re-measure this one.
2. **Two parallel accounting layers with no cross-check** appear independently in recon stage reports (`stats` vs `_reported`, 1.2b) and in digests (score vs body, 1.4). In both, the layer a consumer is likely to read can contradict the authoritative one with no emitted signal.
3. **Launch-contract gaps rediscovered by first-call failure**: `DARK_FACTORY_ROOT` (1.5, 3×), the census verifier's envelope (1.6, 3×), rotation budget (one-off), `iterations.jsonl` (one-off), worktree deps (one-off). The shape matches the codebook's `watcher-capability-envelope` pattern — a contract stated nowhere the consumer can read it, paid for per-session — now observed in five distinct surfaces.
4. **Point-fix confirmation vs. structural fix**: the settled/deferred adjudication gap (1.8) is currently mitigated by prompt prose in both observed rounds; the sightings show the compensation, not a schema fix. Similarly the terminal-write-guard recurrence (§2) shows an undocumented boundary being re-derived per incident.

## 4. Remediation candidates

To be filed via the curator path (plain `submit_task`; curator dedup is the protection — PRD §6.9). Task ids are stamped by the filing stage; rough sizes are S/M.

| # | Candidate | Cluster | Size |
|---|---|---|---|
| R1 | Digest generator: exclude or tag trickle-coder/census meta-sessions from the mining corpus; stop classifying harness-injected task prompts as "User Corrections" (content filter, not position heuristic) | 1.1 | M |
| R2 | Digest generator: emit-time consistency check — nonzero score requires non-empty evidence body; capture the matched substring (guard name/path/symbol) for guard/not-found signals, not just the trigger phrase | 1.4 | S |
| R3 | Recon stage reporting: single authoritative counter set (or emit-time assertion that `stats` and `_reported` agree); resolve `write_journal` `"?"` targets or drop the generic duplicate rows; wire `tokens_used` | 1.2 | M |
| R4 | Graphiti fact extraction: reject/repair facts with unresolved anaphora (no named subject) at extraction time; condition session-start retrieval on the task's domain, not generic recency queries | 1.3 | M |
| R5 | Supervisor rotation prompt/env: export `DARK_FACTORY_ROOT` (or have `watcher-rearm.sh` accept the already-injected project root); state the requirement in the rotation-startup contract | 1.5 | S |
| R6 | Census-session dispatch (verifier **and** synthesis): pre-granted read-only forensic allowlist plus write access to `plans/` and `docs/legibility/`; transcript-path convention stated in the prompt; transcript directories in the allowed working-directory set | 1.6 | S |
| R7 | Consolidator: contradiction check against live same-topic records before canonicalizing a merge; broaden the dedup search radius before asserting "fully resolved"; category assignment gated on the guard's defined trigger, not keyword co-occurrence | 1.7 | M |
| R8 | Prior-round suggestion records: machine-readable resolution-status field (`applied` / `kept-deliberately` / `deferred-open`) so the settled/deferred distinction stops living in judge prompt prose | 1.8 | S |
| R9 | Split orchestrator `merge_queue.py` (739.6KB) and `tests/test_merge_queue.py` (1001.7KB) below the 256KB Read ceiling; consider a repo lint on file-size growth | 1.9 | M |
| R10 | Resume-after-escalation template: make the `iterations.jsonl` read conditional on existence | §2 | S |
| R11 | `resolve_ticket`: return a definitive outcome (or idempotency token / follow-up query handle) on timeout instead of an ambiguous error | §2 | S |
| R12 | `watcher-rearm.sh` return payload: include cumulative rotation-elapsed time vs `ROTATION_HOURS` so budget exhaustion is machine-visible per call | §2 | S |
| R13 | `recon_write_policy.py`: document the terminal-write boundary and either add a corrective path for non-`done_provenance` terminal string fields or record the gap where recon agents will find it (stop the per-incident re-diagnosis) | §2 | S |
| R14 | Worktree provisioning: verify deps installed (`uv sync --all-packages` or equivalent readiness probe) before architect dispatch into a fresh worktree | §2 | S |

Known in-flight work the filing stage should let the curator dedup against rather than pre-filtering here: census-runner CLI-timeout fixes 2951–2953 (run infrastructure, not overlapping the candidates above).

## 5. Codebook dispositions (input to the merger; promote/reject in place, never delete)

| Cluster / finding | Suggested disposition |
|---|---|
| 1.1 digest self-ingestion | New entry (12 sightings, 12 sessions — well past candidate threshold) |
| 1.2 recon self-reporting | New entry; note relation to existing `recon-lifecycle-state-gaps` ("actions conflated with sightings") and candidate `cand-20260719-1` (zero journal entries despite processing) — merger to decide merge vs sibling |
| 1.3 memory injection quality | New entry (two facets in one entry: contentless extraction + unconditioned retrieval scoring) |
| 1.4 digest score/body decoupling | New entry or facet of 1.1's entry — same subsystem, distinct failure (missing vs wrong content) |
| 1.5 DARK_FACTORY_ROOT | New entry; cross-reference `watcher-capability-envelope` (same pattern, different surface) |
| 1.6 census-session envelope | New entry (self-referential: the census's own tooling, now including this synthesis session) |
| 1.7 consolidation surface-matching | New entry; cross-reference `oneoff-2026-07-06` (Mem0 dedup/consolidation collisions — adjacent, distinct) |
| 1.8 settled/deferred adjudication | New candidate |
| 1.9 Read-cap oversized modules | New candidate |
| `resolve_ticket` timeout (§2) | Add sighting to existing `fused-memory-api-traps` |
| Rotation budget (§2) | Add sighting to existing `watcher-loop-harness-mismatch` (adjacent facet: budget invisibility during bounded waits) |
| Terminal-write-guard gap (§2) | Promote — the finding itself verifies recurrence across 2026-07-10/11 and 2026-07-22/23 |
| Remaining one-offs (§2) | Dated one-off entries per existing convention |

## 6. Method notes for the next census

- Segment or exclude legibility-pipeline sessions from the mining corpus (or at minimum tag them) — 37% of this batch's findings are the pipeline observing itself, which will mask target-project trends if left mixed in.
- Phase stamping for non-orchestrated sessions (recon, watcher rotations, legibility tooling) would cut the 25/52 origin-unknown share; the matrix's value to the merge/verify-origin hypothesis depends on it.
- No merge-manifested and one verify-manifested sighting this cycle: either those phases genuinely quieted (the 07-13 survey's fixes landing) or the sample under-reached them. The next census can distinguish by checking sampled-session composition against the class strata.

---

**Handoff summary:** save the above to `plans/confusion-census-2026-07-24.md`, commit docs-only (`git commit --only plans/confusion-census-2026-07-24.md --no-verify`), then run the remaining census stages: curator-path `submit_task` for R1–R14, merger dispositions from §5, and advance `census-state.json`. The memory note `project_confusion_census_2026_07_24_synthesis_blocked_write` records this state for the next session.


## Filed Tasks

_none filed._

## Cost

invoke calls: sonnet miner=360, sonnet verify=1, fable synthesis=1, haiku headroom-probe=1
