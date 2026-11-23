# Forensic re-derivation of the red-chain "elimination" finding (gate 10069)

**Verdict: the challenge holds.** Neither count survives. Of the 15 chains, 8 were verified on top of an unlanded chain that itself went red, 3 were verified while main was red, and in the one sound "wrong conviction" the suspect's content had changed before it passed. No red chain in the records is a flake.

Method, in one line: `merge_verify.merge_sha` is still in git for every chain, so the first-parent walk gives the exact tree tested (each layer is `Merge task/<id>`, parent 2 is the branch tip). Failing tests come from `data/verify-logs/<owner>/`.

Test shorthand used below:

| Short name | Test file | Cause established |
|---|---|---|
| HIST | `tests/scripts/test_merge_gate_history_leg.py` (2 sweep tests) | 7736 at tip `1b28101f` |
| FILING | `test_harness_escalation_filing_identity.py` (2) | 8826 at tip `009377e5` |
| DETRUN | `test_deterministic_runner_project_root_cwd.py` (2) | 5281 at tips `b53d4cf7`, `253f7c15`, `ec33d80e` |
| LOOP | `tests/repo_guards/test_loop_blocking_gate.py` (2) | main red, 11-15 |
| RETIRED | `tests/repo_guards/test_reify_closure_staleness_sweep_retired.py` | main red, 11-17 |
| SUPP | `scripts/tests/test_inline_suppression_ratchet.py` | 6701 at tip `675725bb` |
| UVRUN | `tests/scripts/test_uv_run_venv_isolation.py` | 10242 against 6514 |
| CACHEB | dashboard `test_index_html.py` cache-buster | 5715; 10137 by inference |

The flake ledger agrees these are not flakes: the gate re-ran each in isolation on the failing trees and recorded `fails_in_isolation` (HIST 86, FILING 30, DETRUN 22, LOOP 68, RETIRED 27 rows; no `flake_debt` row for any).

## (i) The 15 chains

All times UTC. "Pred" is the unlanded chain the red chain was stacked on.

| # | Time | Owner: members | Runner, base | a. Member content | b. Tree tested | c. What failed | Class | Conf. |
|---|---|---|---|---|---|---|---|---|
| 1 | 11-15 11:19 | 5189: 5189, 6188, 6179, 5281, 5549 | laptop, main | Constant: all 5 tips equal at their later solos | main `dc2c55f6` + 5, nothing ahead. Main was already red on LOOP, hidden because the single global test command stops at the first failing leg | DETRUN | MEMBER_DEFECT (5281; solo 606871, same tip, on main) | high |
| 2 | 11-15 11:45 | 6188: 6188, 6179 | laptop, main | Constant | main + 2 | LOOP | MAIN_RED | high |
| 3 | 11-16 01:22 | 4991: 4991, 10161, 3638, 3663 | laptop, stacked, spec, depth 1 | Constant: 4991 same tip to its solo and still on `task/4991`; other 3 landed same tips | main + unlanded [7736, 6735, 6573, 6015, 9151, 5918] + 4. `speculative_merge.base_sha` equals the `merge_sha` of verify 608576, which went red 14 min later; that stack never landed | HIST, from 7736 in the base | INHERITED_BASE | high |
| 4 | 11-16 06:37 | 8826: 8826, 8015, 6832 | laptop, stacked | Constant | main + unlanded [3389, 4991, 7736] (verify 610288, red 23 min later) + 3 | HIST from the base, plus FILING, 8826's own | MEMBER_DEFECT (8826; solo 629432) on a base that never landed | high |
| 5 | 11-16 11:57 | 6436: 6436, 6632, 10211, 6787, 8078, 6824 | laptop, stacked | Constant: all 6 landed same tips | main + unlanded [5532, 4991, 7736, 8015, 6832, 5875] (610879, red 4 min later) + 6 | HIST | INHERITED_BASE | high |
| 6 | 11-16 14:39 | 10243: 10243, 8015, 6832 | local, stacked | Constant | main + unlanded [5766, 4991, 7736] (611097, red 5 min later) + 3 | HIST | INHERITED_BASE | high |
| 7 | 11-16 16:22 | 10243: 10243, 8015, 6832 | laptop, stacked | Constant | main + unlanded [10428, 4991, 7736] (611274, red 4 min later) + 3 | HIST | INHERITED_BASE | high |
| 8 | 11-16 18:18 | 5532: 5532, 8015, 6832 | laptop, stacked | Constant | main + unlanded [10422, 4991, 7736] (611465, red 25 min later) + 3 | HIST | INHERITED_BASE | high |
| 9 | 11-17 04:12 | 5281: 5281, 6787, 8078, 6824, 6778, 6641 | local, stacked | 5 constant. 5281 **changed**: `253f7c15` here, `c71d521c` at its pass (fix authored 11-18 07:22), under the same request id `mr-8ddfc972` | The 6 entries ahead landed as built, so tree = main + 6 | DETRUN | Chain: MEMBER_DEFECT (5281; solo 612305 of the same tip failed 37 min earlier). Earlier "5281 passed": CONTENT_CHANGED | high |
| 10 | 11-17 12:32 | 6891: 6891, 6780, 6838, 4915, 6732, 6555 | local, stacked | Constant: all 6 landed same tips | main + unlanded [6836, 4991, 8826, 5281, 6834, 6792] (615518, red 20 min later) + 6 | DETRUN and FILING, from 5281 and 8826 in the base | INHERITED_BASE | high |
| 11 | 11-17 13:39 | 5758: 5758, 4991, 8826 | laptop, main | Constant (5758 landed rebased, own diff identical) | main + 3 | FILING | MEMBER_DEFECT (8826) | high |
| 12 | 11-17 15:17 | 5464: 5464, 4991, 8826 | laptop, main | Constant (5464: same tip failed 17:14, passed 19:54) | main `3759d544` + 3; that base is the commit that made main red on RETIRED | FILING plus RETIRED | MEMBER_DEFECT (8826), with main red at the same time | high |
| 13 | 11-17 21:42 | 7736: 7736, 8826, 5281, 6834, 6792, 6779 | laptop, main | 5 constant; 5281 changed as in #9 | main + 6 | DETRUN and FILING | MEMBER_DEFECT (5281 and 8826) | high |
| 14 | 11-18 19:08 | 10423: 10423, 6514, 10488, 4643, 10429, 8745 | local, stacked | Constant | main + unlanded [8826, 5281, 6838, 6555, 10242, 10422] (622060, red 27 min later) + 6 | FILING, from 8826 in the base | INHERITED_BASE | high |
| 15 | 11-18 19:35 | 8826: 8826, 5281, 6838, 6555, 10242, 10422 | laptop, main | Constant (5281 and 10242 rebased, own diff identical) | main + 6 | FILING only; the scripts leg was green | MEMBER_DEFECT (8826) | high |

Totals for the 15: INHERITED_BASE 7, MEMBER_DEFECT 7 (one of them, #4, on a base that never landed), MAIN_RED 1. FLAKE/INFRA, INTERACTION and UNKNOWN are all 0.

How the earlier counts map onto this:

- **The 7 "last suspect passed" cases** are #5, 6, 7, 8, 10, 14 (inherited base) and #9 (content changed; 5281 was in fact bad).
- **The 2 "last suspect truly failed" cases** are #3 and #4.
  - In #3 the chain's red was inherited, and 4991's later solo failure was the RETIRED test during main red. The lane blocked it as `branch_bug` at 16:38:06 and logged `main_health_red` at 16:38:16.
  - In #4, 8826 was really bad, but the base never landed.
- **The later "failures" of 5189, 6188, 6179, 5549, 5758, 5464 and 4991** were all main-red tests, not member defects.

Main-red proof:

- **LOOP:** it failed in 22 of 22 verifies from 11-15 11:45 to 22:21 (20 solos, 14 different branches). `implicated_commits` names `e8b3ab52aa`, a direct commit to main. There are 10 `main_health_red` rows. It was healed by 10239, which edits `loop_blocking_allowlist.py`.
- **RETIRED:** it failed in 10 of 10 verifies from 11-17 14:44 to 18:44. It was introduced by the docs commits at 14:07–14:38 and healed by `729259fb` at 18:18.

Proof for 7736 as the HIST cause: all 30 trees containing tip `1b28101f` failed HIST, with no exception. Trees holding 4991 without 7736 had a green scripts leg (614951, 615358, 615592). Its next branch commit, `27cd7fa6`, is "list the dispatch-base scene helper as a throwaway-repo history reader". No solo of that tip exists.

Hand re-verified from raw event rows: **608549 (#3), 612323 (#9), 621433 (#14)**.

## (ii) Plain answers

**Recomputed counts under the strict rule.** The rule: base was main or landed as built, main not red on a failing leg, and member content constant by tip or identical own diff.

- **Claim (1), within the earlier 9:** one case remains (#9), and its last suspect, 5281, was truly bad. So 1 of 1, not 2 of 9.
- **Claim (1), re-derived over all 58 red chains with member ids:** 13 are excluded for a base that never landed as built and 4 for main red, leaving 41 eligible. Four are elimination cases:

  | Chain | Last suspect | Truth |
  |---|---|---|
  | 612323 | 5281 | Bad: same-content solo failures 612305 and 619263 |
  | 618484 | 5715 | Bad: same-tip solo 618420 failed on main; passed only after its tip changed |
  | 608576 | 7736 | Bad: by exclusion plus its fix commit; no solo of that tip |
  | 625930 | 10137 | Undetermined: never verified alone |

  Truly bad in 3, undetermined in 1, proven good in 0.
- **Claim (2), within the earlier 15:** 3 chains (#9, 13, 15) have a constant-content verdict for every member on a sound base. Each has a convicted member, so 0 of 3 had no failing member.
- **Claim (2), over all 58:** 7 such chains (612323, 612686, 613201, 617536, 622060, 630804, 637929). 0 of 7 had no failing member.

**How often a red chain is no evidence about its members** (INHERITED_BASE or MAIN_RED):

| Period | Red chains (n) | No evidence | Detail |
|---|---|---|---|
| 11-15 11:19 → 11-17 20:49 (pre-mark) | 45 | 9 | 8 inherited, 1 main red. Two more sat on a never-landed base but also held a bad member; three more ran during main red but also held a bad member |
| → 11-19 04:25 (mark, pre-8528) | 9 | 1 | 621433. Separately, 618460 is the only interaction in the data: 10242 adds a uv-run script, 6514 rewrites the registry test, and each is green without the other |
| after (8528 table), to 11-20 07:10 | 4 | 1 | 637195. One more (643973) was on a re-merged base with cause undetermined. Too thin for a rate |

The remaining red chains break down as:

- **Pre-mark:** 36 MEMBER_DEFECT.
- **Mark:** 7 MEMBER_DEFECT, two of them resting on the 10137 inference.
- **After 8528:** 1 MEMBER_DEFECT, 2 UNKNOWN.

The dominant pre-mark pattern is one bad member re-chained indefinitely: 7736 sat in 30 red trees over 36 hours without a solo, and 8826 in 19.

**Does the lane today mark members of a chain whose red was inherited?** Yes.

- `orchestrator/src/orchestrator/merge_lane/inflight_verify.py::InflightVerifyRun._settle_chain` marks the owner and every link un-chained and requeues on any red tip. It runs when the tip verify completes and makes no check on the base.
- `orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._finalize_inflight` returns on the REQUEUED sentinel before its chain-intact check (`_chain_dead_link` → `_void_and_remerge`). That check therefore protects solo verdicts only, and it knows a base as dead only once recorded; a bad predecessor still in flight counts as live.
- The mark lives in `SpeculativeMergeWorker._next_dispatch_unchained`. It is cleared only in `_retire_item`, and the comment there says a cascade cancel or void keeps it.
- In the records, all 12 red chains stacked on a base that later failed got their verdict 2–40 minutes before the predecessor's.
- The current-era case is 637195: its four members (6960, 9636, 6979, 6984) each then ran alone and passed with unchanged tips.
- Solos are safer: a solo failure on an unlanded base is held until in-order finalize (6161 failed on unlanded 6825 at 11-16 07:35 and landed at 09:03).

**False-failure rate of a verify of a sound tree with constant content.** Window 11-15 11:19 → 11-20 07:10, 214 verdicts: 114 pass, 100 red.

- **Proven false failures: 0.** No red verify on a sound base outside main red has all its members later cleared with the same content.
- **Bracket: 0 of 114 to 2 of 116 (0–1.7%).** The two unresolved are the whole-tree timeout 656320 and the type-error chain 641863. The 95% upper bound on 0 of 114 is about 2.6%.
- **This is after the flake gate.** 6 of the 114 passes carry `merge_flake_suppressed`, so without the isolated re-run the rate would be about 5%.
- **Identical tree verified twice:** only 2 pairs, both agreeing.
- **Where the reds actually fall:** 23 were main red only, 21 were on a base that never landed (13 chains, 8 solos), and 56 were on a sound base with at least one member never cleared unchanged.

## (iii) Where the earlier seat's method was wrong or loose

1. **Identity by request id, not content.** `elim.py::follow` assumes one request id means one content. `mr-8ddfc972` carried two different tips a day apart. Request ids survive "Merge worker shutting down" re-queues (187 such rows in the window), and `merge_queued.head_sha` is null in 622 of 670 rows.
2. **The tested tree was never resolved.** `base_relation` was carried as a label only. 8 of the 9 elimination chains sat on an unlanded chain that went red minutes later.
3. **"Solo failed and applied" was taken as truly bad without joining main health.** Seven of the truth labels are LOOP or RETIRED main-red failures. The `main_health_red` rows sit seconds to minutes after them.
4. **Only verdicts after the chain counted, and unapplied solo failures were skipped.** That discarded 612305, the same-tip solo failure of 5281 just before #9.
5. **`k == 1` was read as "alone on main".** Eight red solos in the window were stacked on unlanded entries.
6. **Failing tests were never read**, only module-level `red_units`. Eight test files explain all 58 red chains.
7. **The "external red rate" (7/46 to 19/58) rests on those 7 chains.** With them reclassified, sound-base, constant-content, no-bad-member red chains number 0.
8. **`falsefail2`'s "14 landed identical to failed content" is mostly main red.** Six are in my window (4 LOOP, 2 RETIRED). Two more on 11-04 are the same RETIRED test on a day with 9 `main_health_red` rows.

## (iv) Could not determine

- **10137:** never verified alone and never landed. CACHEB in 619204 and 625930 is attributed to it by exclusion.
- **643973:** whether 10452 or 5429 causes HIST. 10452's only solo timed out; 5429 has no verdict.
- **641863:** which member causes the type error. I did not open the type log. 5040 changed before passing; 7121 and `fused-recall-review-all-followups` have no later verdict.
- **656320:** whether the timeout is infrastructure or content. There is no re-run.
- **604241:** why this solo passed at 11-15 10:58 on base `e8b3ab52` if LOOP was already red from that commit. No test log is archived for passes.
- **Extra failures beside the deterministic ones** (in 607004, 612686, 619190, 637929): flake or real is undetermined. They changed no verdict.
- **Before 11-15 11:19:** no member ids, so not analysed. The six other "identical content" solo cases from that period are unclassified.
- **The un-chained mark set** is in memory only and emits no event; I inferred it from the next dispatch shape.
- **Syslog** was not consulted; the on-disk verify logs were sufficient.

Outside the brief: tasks 4991, 5189, 6188 and 5549 are `pending` with branch refs unchanged since their main-red block and no later merge request. I did not diagnose why.

Scripts and extracts are in `/tmp/gate-10069/forensics/`: `extract.py`, `build.py`, `fails.py`, `classify.py`, `chain_detail.py`, `preds.py`, `strict.py`, `ffr2.py`, `hand.py`, with `verifies.json`, `fails.json`, `classes.json` and `strict.out`.
