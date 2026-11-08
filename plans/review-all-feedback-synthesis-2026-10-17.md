# /review-all — what to improve after three runs (proposals for discussion)

Written 2026-10-17 by a dark-factory `/team` session (Fable 5.1 lead; five seats: claim verification, task-store dedup, tooling design, data mining, phase 0–5 gap scan; one fable critic on the assembled draft, whose four blockers and ten majors are folded in below — its full report is `S6-critic.md`). Inputs: the three run retrospectives — reify `review-all-reify-20261010` (Phase 6, successor lead), solar_challenge_platform `…-20261013` (Phase 6, resumed lead), solar_challenge `…-20261013` (all phases) — their backing packages, the three run reports and program docs, and the skill at dark-factory main `57fdc61eb2`. Seat deliverables are beside this file (`S1-verify-claims.md`, `S2-task-dedup.md`, `S3-tooling-design.md`, `S4-data.md`, `S5-phase0-5-gaps.md`, `S6-critic.md`).

Three questions: what *could* be improved, what *should*, and how. Tiered: **A** = should, before the next run; **B** = could, worth doing, lower urgency; **C** = retire (already landed or disproved). Each proposal names its evidence, where it lands, and the decision Leo has to make, if any.

## 1. The picture across the three runs

Phases 0–5 (the instrument) are stable. The one cost estimate on record held within 10 % (solar_challenge: 9.27M vs 9–10M); the skeptic bit in every run (weakened share 12–39 %, on different denominators, so not a trend); the critic found 12–22 join errors per run, though no revision was ever re-checked; validation lost nothing in two runs and 15 anchor-shape drops in reify, since fixed (`d1316a47be`). The defects there are plumbing (§A5) and calibration text (§B13).

Phase 6 (the program) is where every run hurt, in the same five ways:

| theme | reify | scp | sc |
|---|---|---|---|
| lead context exhausted mid-program | 940k, handoff | 970k, died with 4 returns in flight | limit on day 4, handoff |
| Phase 6 state in `/tmp` + reaped fleet `result.md`; no resume | F3, F6, F12 | §2.2 (1 h forensics) | §3.6 |
| program-doc tooling hand-rolled per run | `apply_filed.py`, `finalize_wave1.py` | `gen_program_doc.py` | `gen_program.py` |
| free-form seat returns re-typed by the lead | FILED rows parsed by regex | `lead_actions` prose → clerk files | structured block, read by hand |
| spawned-session supervision by hand | 13 h stall, ~100 Monitor re-arms | hand-backs lost to reaping | `find -newermt` |

Three measured facts shape the proposals:

- **The lead arrives at Phase 6 with half its context spent.** scp's lead entered Phase 6 at 526k of a 970k death; a fresh session resuming cost 203k to re-read state (S4 Q3, Q7-9). What exactly consumed the rest cannot be read from the context CSVs (they carry no turn content), but the retrospective says the lead read every hand-back and return in full.
- **Agent-mode streams finish in hours; spawn-mode streams wait overnight.** reify: 2.2–3.9 h launch-to-decompose for the two agent streams vs a 21.7 h median for twelve spawn streams, nine of them ≥ 17 h waiting for pass 1 (S4 Q4; n = 2 vs 12). scp: 2 of 20 decompositions needed Leo (R3 §3.6).
- **Critics refuted lead rulings made without code checks**, twice on one stream (reify S4 OQ-1: the seal created a module cycle; OQ-3: a donor the lead's single-spelling grep missed). Across the 11 streams that counted blockers, every critic reported at least one; the counts are the critics' self-reports and no wave-1-vs-later comparison validates them (R1 `critic-yield.md`).

And two silent-loss bugs in the hand-rolled tooling, one per generator design: reify's `finalize_wave1.py` partitioned the doc on `## FILED` and wrote back only the head (three binding sections gone); sc's `gen_program.py::filed_rows_from_store` filtered on `human_decomposed is not null` and dropped 51 of 188 program tasks, including an adopted pending one — and the package copy never called it at all; the live copy was wired at 19:15 today (S3 §A). Neither was caught by the lead that wrote it.

## 2. Tier A — should, before the next run

### A1. Ship Phase 6 tooling: a program-state home, a rendered program doc, one return schema

**What.** `skills/review-all/scripts/program/`: `state.py` (schema, validator, append-only JSONL, the one store reader), `apply_return.py`, `program_doc.py` (render + `.state/` snapshot + heading self-check + `--roster` + `--handoff`; derives the program-doc path, since no run's filename matched its own `## Next` line), `gen_brief.py` (at spawn time, wave-aware), `clerk_batch.py`, `gate_check.py`, `watch_sessions.py` (archive `result.md` on first sight; stall detection later). Full design, schema, return block and donor code per script: `S3-tooling-design.md` §C–E.

**Design rule.** One source of truth per fact: tasks, held state and the gate from the task store; rulings, amendments, cross-references and dated lead actions in append-only `rulings.jsonl` / `ledger.jsonl`; seat returns kept immutable as audit. The program doc is a render and is never hand-edited; `program_doc.py` refuses to write when a `## ` heading of the existing doc would vanish. Working home `~/.claude/review-all/<run_id>/` (not `/tmp`, not the machine-operated checkout — PRDs are authored there and copied into the checkout at commit time, which also removes scp's 15-hour dirty tree); a `.state/` snapshot committed beside the doc in the same `git commit --only`.

**Store access.** The store is read by a script over the MCP client, as Phase 5 filing already does (`scripts/migrate_metadata_modules_to_files.py::FusedMemoryClient`), never by sqlite: reify's `CLAUDE.md` forbids direct reads while fused-memory is up, and its run recorded exactly that constraint. `find_tasks_by_metadata` (6385, landed and live since 19:37 BST today) returns only id, title, status, priority and `updated_at`, so a render is N × `get_task` (reify N ≈ 528) unless the tool is widened to return full rows.

**Return block.** `review-all-prd-return/1`, one JSON object for agent seats and spawned sessions alike, with `filed[]`, `held[]`, `released[]`, `open_questions[{…, needs_human}]`, `rulings_in_session[]`, `xrefs[]`, `lead_actions[{op ∈ cancel|edge|note|stamp|repath|release|amend, …}]`. `apply_return.py` validates every claim against the store before anything is applied (S3 §D). Held leaves gain `x_seam_wait.release_owner` and, on release, `x_seam_released = {by, at, onto, contract_as_delivered}` — both stay in the `x_` namespace (`docs/task-authoring.md` §Tier-C: silently allowed and findable), no blessing needed.

**Why.** All three runs; the two silent-loss bugs; 17 clerk batches in scp applied with zero errors but every action re-typed by hand; the 13 h stall and reaped hand-backs. This is also what makes A2 (resume, handoff) possible. The first consumer is the next run: solar_challenge is mid-Phase 6 on its own generator and is not migrated.

**Where.** New scripts; `SKILL.md` §Phase 6 and a new §Lead handoff; `deliberation.md` §Recording, §The program doc, §Launching, §Closing; `report-format.md` §Artefact 3 (`.state/`, FILED from store without live status, new `## Held leaves` and `## Ledger`). Line-by-line list: S3 §F.

**Cost.** About seven stdlib scripts with tests; a PRD of roughly 8–12 leaves.

**Decisions.** (a) Delivery path — `/prd` into the orchestrator vs hand-carry. No review-all gate is near firing (6698 deferred; reify #8612 over 61 leaves; sc 772 over 23), so the PRD path fits. (b) Widen `find_tasks_by_metadata` to full rows, or accept N × `get_task` in the script. (c) Who commits a spawn stream's PRD (reify's sessions committed their own; the skill says seats never commit). (d) Commit cadence — see B4.

### A2. Give Phase 6 its own context budget, a fresh lead by default, and a resume

**What.**
1. Phase 6 starts as a **fresh session** anchored on the committed report and record, by default. The phases 0–5 lead ends at the Phase 5 commit by writing `HANDOFF.md` (`program_doc.py --handoff`) and proposing the sitting; the deliberation and program run from a new session (spawned, or opened by Leo).
2. §Lead context budget extends to Phase 6: the lead reads `apply_return.py` digests and `--roster`, never return JSON or full hand-backs; author hand-backs are summarised by script into the questions the lead must rule on.
3. A handoff **trigger** at a stated threshold — 700k is a guess, the limit is not in the data (scp died at 970k with "Prompt is too long") — rather than at the limit; the successor starts from `--roster` and the heading self-check.
4. `--resume <run_id>` gains a Phase 6 meaning: read the state home (or the committed `.state/` snapshot) and the store, diff, continue from the roster's gaps. Never point a brief at a fleet `result.md`.

**Why.** Three of three leads hit the limit, and the one measured boundary shows the lead arriving at Phase 6 with 526k spent. The net on scp is about 225k (526k at the boundary against roughly 100k fresh plus the 203k a resume cost), not the whole boundary. On a large repo the report itself is the obstacle — reify's is 430 KB, which `report-format.md` calls too large for a session to load — so **B5 (split report) is a precondition of A2 there**, not an independent item.

**Where.** `SKILL.md` §Parse invocation, §Lead context budget, §Phase 6, new §Lead handoff.

**Decision.** Accept the handoff at the Phase 5 commit as the default, with B5 as its precondition on repos whose report exceeds ~150 KB (all three so far).

### A3. Per-stream adversarial critic, fold, fold-verifier — and a retro critic on reify's wave 1

**What.** In `deliberation.md` §Launching, between author and commit, for both modes: critic (opus/xhigh, or fable/high for L/high-risk) given the lead's rulings on the open questions → fold → fold-verifier (opus/high) → second fold. A stream that is purely prose or check re-anchoring may replace the fold-verifier with a lead spot-check. The critic brief that worked attacks premises, buildability, invariant enforcement, semantics, seams and manifest polarity (reify `ephemeral/seat-deliverables/S4-critic.md`).

**And a one-off now.** reify's 12 wave-1 PRDs ran no critic and are decomposed into leaves already in reify's queue. If the yield holds at one blocker per PRD, ≥ 12 blockers are dispatchable today. A retro critic pass is ~12 × 0.5M; the alternative is to say why not.

**Why.** The two code-checked refutations of lead rulings are the hard evidence; the per-stream yield (1–5 blockers, 4–19 majors per PRD; fold-verifiers finding 2 blockers on S1, 9 majors on S12, 7 majors and a changed ruling on S2) is self-reported and unvalidated but uniform. A wrong leaf lands in an orchestrator chain; the critic is cheaper than that.

**Cost.** reify's roster figures — 0.3–0.7M per critic, 0.25–0.6M per fold-verifier — are under an unstated counting rule, and S4 measured a 7.5× gap between the two rules in use on one seat. Price A3 in one stated rule at the cost gate; order of magnitude, it roughly doubles a stream's Phase 6 cost.

**Decision.** All streams, or critic-always / fold-verifier risk-gated? I recommend critic always, verifier gated on size L or risk high, and a validation measure in the next run (accepted vs declined critic items per fold; orchestrator outcomes of wave-1-style vs critic'd leaves).

### A4. Deliberating blind is retired

**What.** The skill already offers both never-run instruments whole at the cost gate (the `hotspot baseline` and `/review report` degradation rows); reify's `/review` skip was non-compliance with that row, and its hotspot re-run was offered and declined. The genuinely new rule: a decline does not remove the instruments from the deliberation's inputs — the gap-fill runs **before** the sitting, never after close-out. The "schedule as instrument debt" row goes.

**Why.** reify's gap-fill after close-out found 188 findings (18 high), 8 contradictions with program rulings and a fail-open gate hole, forcing a second sitting and an amended ruling (R1 F1). sc ran both full and its gate held.

**Where.** `SKILL.md` §Cost gate, §Phase 2, §Graceful degradation.

**Decision.** Confirmation only.

### A5. Verdict plumbing in phases 4–5: calibration is recorded, skeptic notes reach the synthesis, join checks are scripted

**What.** Three script changes in `references/orchestration.md` and the Phase 5 digest:
1. **Calibrate's outcome is consumed.** Today `calibrate()` returns outcomes and the sampled finding stays `confirmed` in record, matrix and streams. Two options: record a `calibration_disputes_verdict` beside the finding (the `critic_disputes_severity` pattern — one skeptic overriding another needs a record, and the human rules), and treat a high weakened share in the sample (reify: 5 of 10 highs) as the drift signal the sample exists for — re-verify the high stratum, or caveat every high in the report. Trigger: run the sample when `refuted / n < 0.05` and `n > 10`, not only at zero.
2. **`digestFindings` carries `verdict_notes`** (and the `refuted[]` list) into the synthesis and critic prompts; today weakened findings arrive uncorrected and the critic cannot check them (R Critic "vs fNN skeptic notes"; S Critic "skeptic notes are absent").
3. **Set-membership joins run in the script.** New: a key in two streams, duplicate keys in a matrix cell, a `high` with no stream. Already present and kept: `weakened-hidden` (critic class) and duplicate canonicals across seats (pipeline log). The critic then spends its tokens on evidence-contradicts-conclusion (36 % of the 53 errors) and exonerations (17 of 53).
4. Read the prior-run refutation rate from the record (scp typed reify's as 0.0; the record says 0.003).

**Why.** reify: 5 of 10 sampled highs weakened, nothing consumed it; scp: 2 of 10. All three critics found the same structural join errors by hand and "lesser items" never reached the revision (S5 G1–G3, G7).

**Decision.** Calibration semantics — dispute record, high-stratum re-verify, or both.

### A6. Cross-stream delivered-check churn: an impact scan at decompose time, single-line symbol-anchored patterns, polarity named

**What.**
1. Each decompose return lists the paths and symbols the stream moves, renames or deletes; a script (in A1's `clerk_batch.py`) scans every non-terminal task's `metadata.delivered_checks` for those paths and emits `repath` actions in the same batch (yaml, md twin and task metadata as a complete list — the three-write rule).
2. The `/prd` overlay for review-all programs requires **single-line, symbol-anchored** check patterns; multi-word phrases that wrap across comment lines are rejected (reify S23: 9 checks, 6 vacuous at authoring). The polarity gate's two-sided measurement — fail at the authoring tree, pass after the producer lands (`shared/src/shared/delivered_check_polarity.py`) — stands unchanged.
3. `docs/task-authoring.md` and the manifest schema name the substrate (verdict `PASS`, green by design) vs producer distinction so the gate applies the right rule; `absent_overbroad` is extended to consider files other pending tasks declare they delete (today it checks only the task's own `metadata.files`).

**Why.** reify: six correction rounds, ≈ 50 checks on ≈ 40 tasks in ≥ 10 PRDs; scp: three hazard shapes (over-broad `absent`, sibling-green, moved spelling) (R1 F4, R2 §4). Adjacent tasks: 6037 (polarity structural debt), 6423 (manifest `grep:` evidence format) — extend those rather than file beside them.

**Where.** `deliberation.md` §Launching step 2; `skills/prd` overlay / `docs/task-authoring.md`; `delivered_check_polarity.py`.

### A7. Agent mode by default; a sized second sitting after the author pass

**What.** `deliberation.md` §The program doc sizes a stream by "will its open questions need the human?" rather than "is it design-heavy?". Default `agent` with lead rulings. After the author pass, the `needs_human` open questions across streams (reversing a ruling, cancelling a live branch, moving a board figure, product semantics) go to Leo in **one sitting** in the screens-of-four shape — sc recorded 45 human rulings, so this is ~10 screens, not one. The sitting is a barrier: decompose waits for the slowest author plus Leo's availability, where scp pipelined author→decompose at a 38.6 min median; streams with no `needs_human` question may decompose ahead of it. `spawn` is reserved for a stream whose design choice is itself the human's, runs `/team /prd` with A3's critic, ends after pass 1, and decomposes as an agent seat (spawned terminals stayed open about a day waiting for the lead's pass 2, n = 3).

**Why.** The 10× end-to-end difference (n = 2 vs 12); scp's 4 product questions were genuinely Leo's (two overturned the author's default) but each cost an ad-hoc 0.5M `/team` session (R2 §4); Leo's rulings were most valuable on design forks (S1 R1–R4, S6, S18) — those still reach the sitting.

**Where.** `deliberation.md` §The program doc, §Launching.

**Decision.** The `needs_human` rule list above, and whether non-blocked streams decompose ahead of the sitting.

### A8. Spawned-session supervision and the spawn skill

**What.** `skills/spawn/`: a `--prompt-file` flag on `spawn-claude.sh` (writing the brief to a file is already the documented pattern; sc's apostrophe failures were a wrong escape, the skill documents `'\''`); `/reflect` appended in the trailer (today only the `result.md` hand-back is). In review-all briefs: checkpoint `result.md` after each pass; write `HANDOFF.md` on a context or usage limit; **say when to exit** — an interactive spawned session waits by design, so both Phase 2 sibling sessions idled ~5 h after finishing while the lead waited for an exit that never came; the lead watches `result.md` instead (S5 G13). `watch_sessions.py` (A1) archives `result.md` on first sight; stall detection (status `idle`/`awaiting-input`, no `question`, assistant last, transcript idle > N min) in a second step.

**Why.** R1 F6 (13 h stall; records reaped in 1–2 days; ~100 Monitor re-arms), R2 §2.2 (hand-backs survive only in a dead transcript), R3 §3.8. Done tasks 2287/2934/2285 built the registry and result protocol; nothing covers archiving or stalls (S2 T6).

## 3. Tier B — could

- **B1. Seed G4 from outside the program.** Before authoring and at each wave boundary, scan non-terminal PRDs' Files lists and tasks' `metadata.files` outside the program for each stream's touched paths and add their owners as G4 rows. reify's `codim` clash with pre-existing #6669 was found by a session after S9 had filed (R1 F7). Script on A1's store reader.
- **B2. Held-leaf release protocol** — folded into A1: `release_owner` (default the waited-on stream; the lead when that session is already running), `x_seam_released` with `contract_as_delivered`, a `release` clerk op that adds the dependency, rewrites the description, stamps, and `commit_planning`s the leaf. Two reify releases needed rewording (#8741, #8851). Task 6959 is adjacent.
- **B3. Dedup.** `find_tasks_by_metadata` is live; the key-index cache is retired. Add a one-off `x_finding_key` backfill per project for tasks that carry a finding only in prose (sc 284, 517), and a "known adjacent" list in briefs for the 0.60–0.62 `search_tasks` hits that were never duplicates.
- **B4. Commit cadence.** The constraint is the merge lane's CAS budget: dark-factory `CLAUDE.md` says batch direct commits, because more than `MAX_CAS_RETRIES` (5) landing during one merge's flight blocks that merge — a 26-stream program committing per return is up to 52 docs commits. (reify's #6504 re-verifies came from other sessions' docs commits, not the program's; scp's dirty tree is cured by A1's working home.) Rule to pick: per wave, or per return only when the lane is idle. One decision, one place.
- **B5. Report format: the 60 KB cap is unreachable.** All three reports exceeded it (430 / 176 / 157 KB). Keep the JSON as the record; render the markdown as method + matrix + streams + contradictions + critic (~40 KB) and per-area finding lists as separate files. Precondition of A2 on large repos.
- **B6. Pinned worktree.** Pin under `~/.claude/review-all/<run_id>/tree` (nothing sweeps it; sc's `.worktrees/` copy was removed mid-run and the repo's `.claude` is a dangling symlink); verify the path resolves before Phase 1; state the `git archive <as_of_sha>` fallback in the area prompt. `/review`'s doubled "review-" worktree name is task 6786.
- **B7. One area rule, one home.** Member lists in `review/briefing.yaml` become a hard Phase 0 precondition; the rule is passed explicitly to the Phase 2 siblings and recorded in the record's `method`, not in a scratch script. scp reused the siblings' rule and had 0 area-unmapped errors; reify 5, sc 3 plus five duplicate keys (S5 G4; R3 §3.3).
- **B8. Filing mechanics.** Reconcile before resubmit (a timed-out `submit_task` may have created the task — scp 408); raise the client timeout per `resolve_ticket` call (the client's default is 30 s for every caller); a `combined:<id>` disposition in the Mechanical filings table (sc: 7 of 53 "filed" were curator-combined, so 45 new tasks, not 53); a decompose session takes an edge instead of cancelling a step-1 target that is in progress with a live heartbeat. Measured: 36 % of sc resolves saw a read error, 21 % exhausted the inner retry, one restart failed 55 % of a pass (S4 Q6). Adjacent: 5913/5977, 6705.
- **B9. Fan-out hygiene in Phase 6.** A return under ~100 characters is degenerate on the Agent-tool path too (scp: five decompose seats killed by a session limit returned a 58-character error; 947k tokens of restart waste); size a decompose wave to the remaining allowance or launch after a reset. Record the skill commit in `## Method` and freeze it for the run (reify changed skill text mid-run; its 3.2M Phase 4 re-run folded in the new text and the 15 recovered findings).
- **B10. `## Dropped and degraded` carries operational events** — usage-cap stops, lost filing passes, idle gaps — not only validation drops; scp's section reads "none" six times while its Method notes a lost filing pass (S5 G10).
- **B11. A route for live correctness defects.** A quality review finds correctness defects the orchestrator may meet before the streams land (scp triage briefing, reify α #9062); file them as mechanical tasks at Phase 5, outside the stream queue (S5 G6).
- **B12. Fold-then-cancel.** Cancelling a parent into a stream leaf filed "orphaned: parent cancelled" L1s that became blocking L2s five times in scp; Leo granted a standing rule on 10-17. Put it in `deliberation.md`'s decompose step with scp's other decompose gotchas (planning-mode submit timeouts are not retried; `commit_planning` stamps one sidecar per call) (S5 G14, G22).
- **B13. Cost-gate calibration refresh.** Tokens: phases 3–4 produced 13–23 new findings per 1M tokens across the four runs (reify ≈ 23/M net of its 14.4M of interrupted and re-run work), and source size is not a budget (27–158 tokens per line); area seats ~225k (195–300k), cross seats ~205k, skeptics 90–115k, synthesis ~320k, **critic ~500k** (2.5× the stated estimate); phases 3–4 wall clock 4.4–7.2 h, not 1.5–2.5 h, with phase barriers and usage waits the cause. **Human hours**, which the gate must post and no run calibrated: scp ≈ 1.5 working days plus a day of rulings; sc 4 attended days, ~35 terminal sessions, 45 human rulings; reify ≈ 2.5 days plus a second sitting. State the token-counting rule in `## Method` (scp's filing seat is 180k under one rule and 1.34M under the other).
- **B14. Synthesis prompt hardening.** Exonerations (17 of 53 critic errors) and duplicate-mechanism (8 of 53) get explicit checks in synthesis jobs 4–5; the revised synthesis gets a cheap re-check of the critic's listed items; the area-seat severity rule gains "is the wrong version caught, and by what" (all four downgrades were high→medium on that basis, 0 disputed) (S5 G7–G9, G18).
- **B15. Label synthesis counts `measured at <as_of>`** and require a `/prd` session to re-measure any count a leaf's done-criteria rest on (reify: "~700" cites were 2,480; "67 tests" were 19) (R1 F11).
- **B16. Python snapshot on a mixed repo** leaves TS/manifest files unmeasured (scp: 153 files, 16k lines; one area had no ranking instrument); the overlay names a second instrument or the report says the area is unranked (S5 G5).

## 4. Tier C — retire or correct

- **Single-package metrics instrument**: task 6921 is done (`a5712a5565`); no shim.
- **Key-index cache (R1 F9)**: superseded by 6385, live. A session that lacks the tool reloads its MCP roster; do not restart the server.
- **"Skip with debt recorded as a default" (R1 F1)**: the hotspot re-run was offered and declined; A4 is phrased accordingly.
- **Counts the records qualify**: sc's seats are 24 stated / 23 in the workflow record; sc "53 tickets filed" is 45 new tasks; sc's `refuted` stayed 1 in the committed record; the four refutation rates sit on different denominators (99 / 348 / 64 / 203) — compare the weakened share instead (S4 Q1, S5 §2).
- **Task 7335** is a reify task unrelated to review-all; the dark-factory companion in reify's gate is 6856.

## 5. How — delivery

1. **Skill-text amendments** (A2 text, A4, A7, B4, B7, B9–B12, B13, B15, parts of A3/A6/A8): docs-only commits on dark-factory main, batched into one or two `git commit --only` — the shape every prior amendment took.
2. **Script work with tests** (A1, A5, A6 scan, A8 spawn changes, B5 render, B6): a PRD, `review-all-program-tooling`, decomposed through `/prd` and run by the orchestrator. No review-all gate is near firing, so the queue has time.
3. **Cross-skill changes** (A6 polarity gate and task-authoring; A8 on `skills/spawn`; B8 on the filing client): separate small tasks, each citing the adjacent task (6037, 6423, 6959, 5913/5977).
4. **One-offs**: the reify wave-1 retro critic (A3); `x_finding_key` backfills (B3).

No proposal above has an existing task covering it fully (S2); adjacent tasks are named where they exist.

## 6. The discussion agenda

1. **Delivery path** — PRD through the orchestrator vs hand-carry (timing is not a constraint).
2. **Fresh lead for Phase 6** at the Phase 5 commit, with the split report (B5) as its precondition.
3. **Critic scope** — critic-always / verifier-gated — and the retro critic on reify's 12 wave-1 PRDs.
4. **Agent-mode default** and the sized second sitting: the `needs_human` rule list; whether unblocked streams decompose ahead of it.
5. **Store access for the render** — widen `find_tasks_by_metadata` to full rows, or N × `get_task` in a script.
6. **Commit cadence** under the merge lane's CAS budget — per wave, or per return when idle.
7. **Calibration semantics** — dispute record, high-stratum re-verify, or both.
8. **Confirm** that deliberating blind is retired (A4).

## Appendix — team roster and outcome

| Seat | Model / effort | Tokens | Min | Outcome |
|---|---|---|---|---|
| S1 verify claims vs tree | sonnet / medium | 165k | 11 | 25 claims classified; 2 already landed (6921, 6385), 1 partial (polarity gate), rest open |
| S2 task-store dedup | sonnet / medium | 163k | 16 | no theme fully covered by an existing task; adjacent tasks named |
| S3 tooling design | opus / high | 239k | 32 | comparison of 3 generators, 6 defects quoted, schema + script set + skill-text list |
| S4 data mining | sonnet / high | 227k | 27 | cross-run calibration table, context curves, 15 unreported findings |
| S5 phase 0–5 gaps | sonnet / high | 293k | 37 | 22 defects not in the retrospectives (3 high), 11 qualifications |
| S6 critic of the assembled draft | fable / high | 257k | 22 | 4 blockers, 10 majors, 14 minors, 6 missing — all accepted after the lead verified the blockers against live state |

Dropped or unverified: no lane failed. Not independently checked by the lead: S4's per-turn context attribution (the CSVs carry no turn content) and S5's reading of program-doc sections the lead did not open. `get_tasks` on dark-factory exceeded 16 MB, so S2's recall rests on `search_tasks` plus a read-only sqlite title scan of dark-factory's own store.
