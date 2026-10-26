# D7 - consistency review of the amended /review-all skill (2026-10-19)

Worktree: /home/leo/src/dark-factory/.claude/worktrees/review-all-amend-20261019. Read-only. Files: SKILL.md (S), references/deliberation.md (D), references/report-format.md (R), references/orchestration.md (O). Line numbers are the worktree files as they stand.

Verdict at the end: **needs fixes** (4 real defects, 8 minor).

## 1. Cross-references resolve - FAIL (3 dangling targets, 1 wrong target)

All `§Heading` references between the four files and into `skills/spawn/SKILL.md`, `skills/_shared/filing-the-trigger-chain.md` (§0-§5), `skills/team/SKILL.md` (§Fable, §4), `docs/task-authoring.md` (§Tier-C), `CONTRIBUTING.md` §6, `docs/code-quality.md` §Relationship to the design invariants, `CLAUDE.md` §Working in the main checkout (MAX_CAS_RETRIES (5) is there) resolve to a heading. Bold-run-in "headings" (Slices, Rotation, Read shortfall) are not `##` headings but are the established way SKILL.md cites them; fine.

Failures:

1. D:54 `plans/review-all-program-tooling-prd.md` §Return block - **the file does not exist** in the worktree, in the main checkout's plans/, or in git history. The seat (D2 item 7) cited `plans/review-all-feedback-synthesis-2026-10-17/S3-tooling-design.md` §D; the lead's patch re-pointed it at a PRD that has not been written. S:153 ("The program-tooling PRD adds `program_doc.py --handoff`") is a forward reference of the same kind, harmless as prose, but D:54 is a pointer a seat is told to follow for the field shapes of `review-all-prd-return/1`.
   Fix: either say in D:54 "field shapes: the program-tooling PRD, not yet written - until it lands, the keys listed here are the schema", or commit the S3 doc and cite it.
2. S:66 `plans/review-all-feedback-synthesis-2026-10-17.md` §B13 - the file exists only as an **untracked** file in the main checkout (B13 is there, line 138), not in the worktree and not committed. After the wave commits, the cite dangles in any fresh checkout. Fix: commit the synthesis plan with the wave, or drop the cite and keep the figures (they are quoted in full in S:66 already).
3. D:61 "`outcome: in-progress` ... per `skills/spawn/SKILL.md` §Result-handback" - the heading exists, but its outcome enum is `done|blocked|abandoned|handed-off` (skills/spawn/SKILL.md, Result-handback bullet 2); `in-progress` is not defined there. The citation attributes to spawn a value it does not have (Leo's A8(i) ruling asks for that change to the trailer/archiver; it has not been made). Fix, one line: "a checkpoint carries `outcome: in-progress` (to be added to the spawn trailer's enum, `skills/spawn/SKILL.md` §Result-handback; task needed), only the final write a terminal outcome".
4. S:159 ends "A wave's docs changes are committed once, through the lead (`references/deliberation.md` §Launching `/prd` per stream)" - the rule's home is D `## Commit cadence`, not §Launching. Fix: replace the cite with `§Commit cadence`.

## 2. Numbers quoted in more than one file - FAIL (1 mismatch)

| Figure | Where | Verdict |
|---|---|---|
| 526k / 970k / 203k | S:139 only | single site |
| ~700k handoff threshold | S:155 only | single site |
| 0.05 / 5 % | S:133 (0.05, ratio form), O:102, 512-514 (code: `>= 0.05`), 758, 773, 788 (5 %) | agree (same value; two spellings, one per file family) |
| 30 % / >= 2 highs | S:133, O:104, 788 | agree on numbers (see 5.x on "weakened" vs "disputed") |
| 430 / 176 / 157 KB | S:143, R:5 | agree |
| 40 KB cap | S:143 ("~40 KB digest"), R:3 ("capped near 40 KB") | agree |
| **60 KB cap** | S:143 "the old 60 KB cap" ; R:5 "430 / 176 / 157 KB **against a 60 KB cap**" | **MISMATCH** - R:3 says the cap is now ~40 KB, R:5 still calls 60 KB "the cap". Fix: R:5 "against the old 60 KB cap". |
| 60 KB `args` (vs 820 KB) | S:107, S:176, O:21, O:780 | agree (different quantity, same value everywhere) |
| 5411 | D:66 only ("30-day shadow is ruled") | single site |
| 6385 | S:145 only | single site |
| 7150 | **not present in any of the four files** | nothing to compare; if it was meant to be cited (brief lists it), it was not added |
| 4.4-7.2 h; critic ~500k (494k, 2.5x) | S:66 only; O's routing text says critic 100-200k, so 2.5x of the upper bound = 500k | consistent |
| human-hours line | S:66 only; matches B13 (scp 1.5 days + a day; sc 4 days / ~35 sessions / 45 rulings; reify 2.5 days + second sitting) | single site, matches source |
| 5 of 10 / 2 of 10 | S:133, R:88, O:788 | agree |
| 17 of 53 (critic errors in exonerations; 8 duplicate-mechanism) | O:697, O:790 | agree |
| ~10 h no merges; MAX_CAS_RETRIES (5) | D:66 only | single site |
| 0.003 vs 0.0 (reify refutation rate) | O:773 only | single site |

## 3. Vocabulary - FAIL (1 real, 3 minor)

- `calibration_disputes_verdict`: **R:61 (record schema) says `{to: weakened, note}`; R:88 (same file) says `{to: weakened | refuted, note}`; O:`recordOutcome` and O:788 set `to` to `weakened` or `refuted`; S:133 says only "a sampled finding the fresh skeptic weakens".** This is D2 judgement-call 3, still open. Fix R:61 to `{to: weakened | refuted, note}` and S:133 to "weakens or refutes" (the code and the drift count already treat both).
- `calibration.high_stratum` / `high_stratum_reverified`: R:67-ish schema `{reverified, confirmed, weakened, refuted, lost, outcomes}` = O code's `{reverified, ...tally, outcomes}`. Agree. S:133 and R:88 use the same names. PASS.
- `synthesis.join_errors_mechanical`: S:133 (no field name, "run in the script"), R:88 and R schema, O (`join_errors_mechanical`). Name agrees. The three checks are worded three ways: S "a `high` with no stream"; R "a high finding in no stream"; O "a high **structural** finding in no stream" (mechanical highs are not streamed, FM21). Minor; fix S and R to "a high structural finding".
- `yellow_flag_sample` shape: the script now returns an extra boolean `drift` (O, `calibrate()`); R:Method yaml lists `{sampled, strata, confirmed, weakened, refuted, lost, outcomes}` without it. Minor; add `drift: <bool>` to R's yaml line.
- `review-all-prd-return/1`: D:54 only; `op` list includes `release` and `amend`, and the `moves` field (the D2 "no field for A6(a)" gap) is now present. PASS (modulo 1.1).
- `x_seam_wait` {stream, contract, release_owner}, `x_seam_released` {by, at, onto, contract_as_delivered}: D:58 and R (held-leaf bullet) agree on names and on the release steps (D's four writes include "description rewritten to the contract as delivered", which R expresses as the `contract_as_delivered` field; compatible, R does not mention the description rewrite - not a conflict).
- `outcome: in-progress`: see 1.3 (not in spawn's enum).
- `~/.claude/review-all/<run_id>/`: S:153, S:155, D:54 - same spelling. PASS.
- `.state/`: **S:155 only** ("committed `.state/` snapshot beside the program doc"). D:66 says "the state snapshot" is committed with the wave, but no file says what it is, where it is written, or who writes it, and R Artefact 3 does not list it. A dangling concept (presumably the tooling PRD's). Fix: one sentence in R Artefact 3 ("`<program doc>.state/` - the snapshot of the task-store view the lead refreshes at each wave commit; until the tooling PRD lands it is optional") or drop the `.state/` clause from S:155 and D:66.
- `--full`, `--resume`: S:4 (hint), S:27, S:32, S:39, S:55. Consistent: `--full` excludes `--plan`, does not imply `--yes`, keeps the headroom rule (S:39), and S:55 ("`--yes` or `--full`: post it and proceed") agrees.

## 4. A rule stated in two files - PASS with 2 notes

| Rule | Homes compared | Agree? |
|---|---|---|
| Commit cadence | D `## Commit cadence` (home): once per wave via lead/clerk, `--only`, until 5411 then merge queue. S:159: "seats author, **the lead commits, seats decompose in seam order, the lead commits again**" and "A wave's docs changes are committed once". D:56 (decompose step): "into the wave's one commit". | **S:159 contradicts itself and D**: "commits again" = two commits per wave vs "once". Fix S:159: "seats author, seats decompose in seam order, the lead commits the wave once (§Commit cadence)" (and 1.4). |
| Held-leaf key shapes | R Artefact 3 (home), D:58 (references, restates only names) | Agree. |
| Report split | R Artefact 1 (home: digest ~40 KB, `.findings/<area>.md` + `repo.md` + `cross-area.md`, §5 and §6 moved); S:143 (digest ~40 KB, `.findings/<area>.md`, committed with the JSON and snapshot) | Agree (S omits repo.md/cross-area.md - fine). Only R:5's "60 KB cap" is off (section 2). |
| Calibration mechanics | O (home: trigger, sample, recordOutcome, drift, stratum); S:133 restates trigger and drift; R:88 restates record shape | Numbers agree. Wording differs: S "weakened share of >= 30 %" vs O "disputed (weakened or refuted)"; S "n > 10" (of what?) vs O code `confirmed + weakened > 10` and FM4 "more than ten surviving findings". Fix S:133: "a disputed share (weakened or refuted) of >= 30 %" and "with more than ten surviving findings". |
| Phase 2 never-run rule | S §Phase 2 (home); S degradation rows 2, 3 and "Phase 2 siblings not run" cite it | Agree: the gap-fill runs before the Phase 6 sitting, never after close-out. D does not restate it. |
| Spawned-session lifecycle | D:61 (home); S:159 "spawned interactive sessions for `spawn` streams" | Agree. |
| "A brief never points at a fleet result.md" (S:155) vs "the lead watches result.md" (D:61) | Not a conflict: S forbids a brief *citing* a path the registry reaps; D has the lead read the live file and keep the durable copy in `inbox/`. State-home inbox is the durable path in both. |

## 5. Contradictions with UNCHANGED text - FAIL (2) / PASS (rest)

1. **Frontmatter description (S:3)** still ends "state the cost, Phase 2 included, and get a go before phase 2". `--full` (and `--yes`) now proceed without a go. Fix: "...and get a go (or run with `--full`/`--yes`) before phase 2."
2. **Phases 3-4 roster table (S:122-129)** lists the calibration skeptic (10) but not the new high-stratum skeptic (O routing table has it: sonnet/high, one per remaining confirmed high, only on drift). The Cost gate prices "the roster in §Phases 3-4", so the new seat class is invisible to the gate and to the diagram's `~20+4 seats`. Fix: add a row `| high-stratum skeptic | one per remaining confirmed high, only when the calibration sample drifts | sonnet | high |`.
3. Degradation table vs §Phase 2: rows 2, 3 and 11 now say the gap-fill runs before the sitting; §Phase 2 says the same. PASS.
4. `## The program doc` "a design-heavy stream becomes `spawn`" (D:47, unchanged by ruling A7) vs the new spawn bullet "END after pass 1": not contradictory (a design-heavy stream's interactive discussion happens inside the author pass; critic/fold/decompose then run as agent seats). Note for the reader only: D:47 does not say a spawn stream's later passes are `agent` seats; D:61 does. PASS.
5. O routing table vs `Reverify high stratum`: table row present, meta.phases entry present (between Cross-area and Synthesize), script calls `phase('Reverify high stratum')`. PASS. FM16 now says "Calibrate (with its high-stratum step)". PASS.
6. Failure modes numbered 1-21, contiguous; every "failure mode N" cite (1, 4, 5, 11-19, 21) points at an existing item. FM4 -> FM19 and FM19 -> FM4 cross-cites agree. PASS.
7. S:133 says the sample is "beside the cross lenses" and returned as `yellow_flag_sample`; O awaits it before the digest so synthesis/critic see disputes. Compatible.
8. D intro / Commit cadence vs Phase 5's own `git commit --only` (S:143): Phase 5 is the run's report commit, before Phase 6 waves; no conflict.

## 6. JS blocks in orchestration.md - PASS

Two fenced ```js blocks (args schema; workflow script), each wrapped in `async function f(){...}` with `export` stripped, `node --check` (node v22.23.2): both **OK**. Extracted to `.../scratchpad/d7js/b1.js`, `b2.js`. Also read: `mechanicalJoinErrors` runs on short ids before `mapIds` (the only `mapIds` call is in the return), `canonicalOf`, `known`, `idOf` are defined before use, and `calibration` is initialised before `reverifyHighStratum` writes it.

## 7. Deletions with no new home - PASS (1 minor)

Every removed sentence is re-stated or superseded: the `--yes`-only cost-gate wording (now `--yes` or `--full`); the zero-refutation trigger (now 5 %); "~60 KB human-facing form" (now 40 KB digest + findings files); "instrument debt in the program doc" (replaced by the gap-fill rule, S:114 + degradation rows); Phase 6 "Close the run turn by proposing deliberation" (now handoff-first, S:151-159); the single-commit sentences in D (now `## Commit cadence`); `x_seam_wait` two-field shape (now three fields); the old `## High findings` / `## All findings` per-area-in-digest text (moved to findings files). One loss: the old held-leaf sentence "released by the owning stream's session" is now "release owner" (more general, covered).
Minor: the removed D "Commit it with `git commit --only`" for the program doc is covered by `## Commit cadence`, which lists the program doc. No orphaned rule.

## Verdict: needs fixes

Must fix before commit (real defects):
1. R:61 `calibration_disputes_verdict` schema `to: weakened` -> `weakened | refuted`; S:133 "weakens" -> "weakens or refutes" (and "disputed share").
2. S:159 "the lead commits, ... the lead commits again" -> one wave commit; cite `§Commit cadence`, not §Launching.
3. D:61 `outcome: in-progress` is attributed to spawn SKILL.md §Result-handback, whose enum lacks it - reword or amend spawn (A8(i) is still undone).
4. D:54 cites a non-existent `plans/review-all-program-tooling-prd.md` §Return block; S:66 cites an uncommitted synthesis plan. Commit the cited docs with the wave or reword.

Should fix (minor): R:5 "60 KB cap" -> "old 60 KB cap"; S:3 frontmatter "get a go" -> add `--full`/`--yes`; S roster table missing the high-stratum skeptic row; `.state/` snapshot defined nowhere (S:155, D:66); `drift` missing from R's `yellow_flag_sample` yaml; "high structural finding" wording in S and R; S:133 "n > 10" wording; `7150` listed in the brief but cited nowhere.
