# /review-all run feedback — review-all-solar_challenge_platform-20261013

Written 2026-10-16 by the resumed lead session (Fable 5.1) that completed Phase 6 after the original lead hit its context limit. Input to a dark-factory team session synthesising feedback across runs before changing `skills/review-all`. Backing data and provenance: `plans/review-all-feedback-solar_challenge_platform-20261013/` (read `PROVENANCE.md` first; it separates copied ephemera from pointers to durable records).

Scope: this report is about **Phase 6** (deliberation → program doc → `/prd` per stream → trigger chain) and the lead's operating model. Phases 0–5 ran on 2026-10-13/14 under the original lead and are recorded in the run report; I did not re-review them beyond what the resume required.

## 1. What happened

| When (local) | Event |
|---|---|
| 2026-10-13 07:30 → 2026-10-14 02:30 | Phases 0–5 (one lead session, b1f15ebe). Report + record committed e4f2a2b at 12:46 on 10-14 after the sitting. |
| 2026-10-14 morning | Deliberation sitting with Leo: 26 streams, 20 rulings. Program doc 207d800 at 13:37. |
| 2026-10-14 13:50 → 14:59 | Author pass batch 1: 13 agent-mode seats (opus/high), commit 363f63c. |
| 2026-10-14 15:09 | **Five decompose seats died with "You've hit your session limit · resets 5pm"** (S14, S17, S20, S21, S23). Relaunched 15:55–16:01 after the reset. |
| 2026-10-14 14:24 → 17:41 | Decompose passes for 16 agent-mode streams + S3/S5/S6; 7 clerk batches; 6 program-doc commits. |
| 2026-10-14 16:46 → 17:47 | The lead's context (970k tokens at the last successful turn) overflowed: every turn from 16:46 returned "Prompt is too long" while four decompose seats (S13, S10, S1, S4) and three clerks finished and reported into a dead context. |
| 2026-10-15 13:46 | Resume (this session, `/team`). State rebuilt from the FILED table, the dead session's `/tmp` scratch, subagent transcripts and fleet `result.md` files. |
| 2026-10-15 15:03 → 20:30 | S1/S4/S10/S13 recorded; five PRDs committed; six decompose seats (S2, S7, S11, S12, S16, then S15); nine clerk batches; gate 1065 verified (67 deps = every high leaf) and committed by Leo at 20:1x. |
| 2026-10-15 21:20 → 2026-10-16 16:00 | Four `/team` ruling sessions for the product questions the author seats raised; results applied through the generator. |

Totals for Phase 6 (both lead sessions; `metrics/seats-summary.md`): 25 author seats (13 agent + 12 spawn), 26 decompose seats (opus/high), 19 clerk batches (sonnet/medium), 1 Phase-5 filing seat, 4 spawned ruling sessions (each a 3–4 seat `/team`), 1 merge re-drive seat. Approximate subagent tokens: 11.7M decompose (dead session, 23 seats incl. 5 limit-killed) + 3.7M decompose (resume, 6) + 7.3M author (agent mode) + 8.8M clerks + ruling sessions ≈ 2.2M. Wall clock: about 1.5 working days from sitting to a complete FILED table, plus a day for the four rulings.

## 2. What broke

### 2.1 The lead has no context budget in Phase 6

The skill budgets the lead's context for phases 3–5 ("the lead never reads seat prose; it reads digests and summaries that scripts produce") and then says nothing for Phase 6. The dead lead read every author hand-back in full (25 × 7–9 KB), every decompose return (20 × ~9 KB of JSON with prose `lead_actions`), and inlined a ~4 KB brief into every seat prompt. `metrics/lead-context-dead-lead-b1f15ebe.csv` shows the context passing 800k during the decompose pass and dying at 970k with four returns in flight.

**Proposal.** Apply the phase 3–5 discipline to Phase 6: seats write their return JSON to a file under the run directory and return a one-line digest; the lead reads a scripted roster (stream, filed ids, gate deps, held leaves, action count), never the JSON. Author hand-backs are summarised by a script into the questions the lead must rule on; the full text is only opened per question.

### 2.2 Phase 6 has no resumable state

`--resume <run_id>` covers the Phase 3–4 workflow journal. Phase 6's state lived in four places the skill never names: the committed FILED table; the lead's `/tmp` scratch dir (`filed/*.json`, `prd-results/*.json`, the generator); the subagent transcripts (the only copy of seat returns that arrived after the lead died); and fleet session `result.md` files, which are reaped — two were gone within a day, and the 2026-10-14 hand-backs now exist only inside the dead transcript. Rebuilding took about an hour of forensics and a `/team` invocation; the state itself was intact.

**Proposal.** A `program-state/` directory under the run's plan directory (or the run's scratch dir mirrored there at each commit) holding `program-state.json` with per-stream status (`authored | committed | decomposed | recorded`), the seat returns, the author hand-backs copied in at receipt, and the lead-action files. Add `--resume-program <run_id>`: read the state, diff against the FILED table and the task store, finish from the gap. Never point a later brief at a fleet session file.

### 2.3 The program-doc generator is the right design and is not in the skill

The dead lead wrote `gen_program_doc.py` ("generated from the record, rulings.json and filings.json; the lead edits nothing by hand after") and `gen_prd_briefs.py`. Both are exactly right: the program doc was regenerated 11 times across the two days and never drifted from its inputs, and applying 26 FILED rows and a dozen later rulings was mechanical. But the generator was hand-rolled per run, lived in `/tmp`, had the project's stream table and S8 routing hard-coded, and kept its ledger of cancellations and releases as a hard-coded prose string that I appended to five times. The skill text still says the program doc is "written once" and the lead "applies the returned rows".

**Proposal.** Promote the generator to `skills/review-all/scripts/program_doc.py` with a stable input schema — `rulings.json`, `filings.json`, `streams.json`, `amendments/S*.json`, `filed/S*.json`, `ledger.json` (dated actions: cancel/amend/stamp/edge/release/ruling) — and make the skill say the program doc is regenerated from those inputs after every step. Copies of the two scripts are in the package.

### 2.4 Usage limits inside a fan-out

Five concurrent opus seats were killed at 15:09 by the account's session limit, returned a 58-character error as their only output, and were relaunched 45 minutes later. The skill's cost gate asks the lead to check weekly headroom before Phase 2; it does not mention per-session limits during Phase 6 fan-outs, and `cleanSeat`/`degenerateSeat` validation exists only in the Phase 3–4 workflow. The dead lead did detect and relaunch them by hand.

**Proposal.** Treat a seat return under ~100 characters as degenerate in every phase (the Agent-tool path has no schema check), and say in Phase 6 that a decompose wave should be sized to the remaining session allowance or launched after a reset.

## 3. What worked and should be codified

- **Clerk seats from written action files.** Seventeen sonnet/medium batches across both days applied cancellations, edges, key stamps, notes, held-leaf releases and delivered-check repairs with zero errors and a full audit trail (`lead-actions/`). The one correct skip (785: a branch with commits) came from a rule written into the file. The remaining cost is that each decompose seat returned `lead_actions` as prose and the lead re-typed them into the batch file.
  **Proposal:** decompose seats return `lead_actions` as structured operations (`{op: cancel|edge|note|stamp|repath|release, task, target, text}`) and the clerk applies the file as returned; the generator's ledger consumes the same records.
- **Commit per seat return, never across concurrent seats.** Six decompose seats stamping sidecars concurrently caused no conflicts because each stream's sidecar was committed the moment its seat returned. The dirty tree the dead lead left for 15 hours (five uncommitted PRDs) coincided with the orchestrator being in a restart drain until they were committed.
- **Gate integrity check.** Comparing the gate task's dependency list with the union of every stream's declared `high`/`critical` leaves is a ten-line script (`briefs/resume-session/` has it inline in the transcript); it found exact agreement (67 = 67) and should run at closing.
- **Per-stream "PRD authored" memories** written by the author sessions (holds on 582/584, the ProviderMode home, 466's overlap) were the first thing that oriented the resume. Keep them; the skill could name them as a Phase 6 artefact.
- **Parallel decompose with concurrent id allocation** worked: seats noticed taken ids (1209→1210, 1231–1244 interleaved) and recovered without instruction beyond "get_task/search before retrying".
- **Routing held.** No decompose seat needed a redo (opus/high, 230–280k tokens, 30–50 min each); clerks at sonnet/medium were right; the four ruling sessions ran as `/team` with 3–4 sonnet seats each and produced rulings Leo accepted.

## 4. Hazards the seats found that the skill should ask for

- **Cross-stream delivered-check hazards.** Three shapes appeared: (a) an over-broad `expect: absent` pattern that also matches a definition another stream keeps (1205's `current_finance\(` over the whole package matched the `def` in `config/store.py` and would have stayed red forever; caught by `commit_planning`'s warning); (b) a check that goes green when a *sibling* leaf lands before its own task (459's lifecycle checks, 460's `bill_account` check — vacuous for the owning task); (c) a check whose pinned spelling another leaf moves to a different file (493's `CONSENT_CHECK_FAILED` moved by 295 → removed-path wave on 493's dependents). The S2 seat found (c) by reading sibling task text.
  **Proposal:** the decompose brief requires a scan of every filed task's `delivered_checks` against the leaf's deletions, moves and renames, and `commit_planning`'s polarity gate could evaluate `absent` patterns against the files other pending tasks declare they delete.
- **Product questions surfacing at author time.** Four of the six spawn-mode streams carried a question only Leo could answer (S15 Q3, S16 Q1, 582 vs 617, 382), each ruled later by a `/team` session at ~0.5–0.6M tokens. The questions were genuinely new (two overturned the author's default), so the cost was warranted, but the shape was ad hoc.
  **Proposal:** a second deliberation screen at the end of the author pass — all open product questions across streams, in the `references/deliberation.md` shape — before any decompose seat launches; and the ruling-session brief shape used here (`briefs/spawn-briefs/rulings-2026-10-15/COMMON.md` in `~/.claude/spawn-briefs/`) as the template.
- **Held-leaf releases leave `x_seam_wait` behind.** Released leaves (1205, 1117, 1116, 1118, 1133, 1119) keep the metadata; nothing reads it, but a future tool might. Release should clear it or stamp `x_seam_released_by`.
- **The generator's trailer.** Dated lead actions were appended as prose five times; by the end it was a 400-word paragraph. This is the `ledger.json` case for §2.3.

## 5. Calibration to record in the skill

The skill's calibration is the first dark-factory run (26 streams, "one sitting, six `/prd` seats in wave 1"). This run, also 26 streams, measured:

| Item | Measured |
|---|---|
| Author pass | 13 agent seats (opus/high) in one wave ≈ 7.3M tokens, 70 min; 12 spawn sessions over the afternoon |
| Decompose seats | 26 at opus/high; 230–280k tokens and 30–50 min each; none redone; 5 killed by a session limit and relaunched |
| Clerk batches | 19 at sonnet/medium; 80–170k tokens and 5–15 min each; 0 errors |
| Lead context | dead at 970k after ~34 h of session time; the resume used 585k for the second half |
| Phase 6 wall clock | sitting 10-14 morning → FILED complete 10-15 16:00 (with a ~20 h gap for the death) → rulings applied 10-16 16:00 |
| Product rulings after authoring | 4, each a `/team` session of 3–4 sonnet seats ≈ 0.5–0.6M tokens |
| Pre-existing tasks disposed | 24 cancelled as accepted, 11 amended in place, 8 key-stamped, ~45 cross-PRD edges |

## 6. Smaller observations

- Two decompose seats (S10 and S1) reported that the author hand-back path in their brief no longer existed by the time they ran; the dead lead's briefs pointed at fleet session files (see §2.2).
- `search_tasks` step-2 matches at 0.60–0.62 were almost always adjacent-not-duplicate (298, 899, 342, 228); the threshold is right but the seats spent effort reading each. A "known adjacent" list in the brief would save it.
- The skill says the lead "commits every PRD in one `git commit --only`" after the author pass; with spawn-mode authors finishing hours apart, the dead lead committed the agent-mode batch and never got to the spawn-mode five. Commit each PRD as its hand-back arrives.
- The merge re-drive seat found the orchestrator had already restarted and landed one of the four branches itself; the skill should say the lead re-reads live orchestrator state before acting on a stale "drain" observation (this is the existing `unblock-stale-escalation-verify-live-state` memory, applied to the review-all lead).
