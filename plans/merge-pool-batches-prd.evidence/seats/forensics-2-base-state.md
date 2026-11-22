# Follow-up: conditional evidence of a verdict by base state (gate 10069)

**Leo's model fits the records.** A red on an already-green base had a defective own member in every determinable case (43 of 43). A red on a base that later went red carries no information about its own members. No green verdict was ever recorded on top of an entry that returned red.

Window 2026-11-15T11:19Z to 11-20T07:10Z: 214 verdicts, 114 green and 100 red.

**Correction to my earlier report.** There was a third main-red episode: the direct commit `6d1efef4da` (`link_heal.writes true`, 11-16 06:18) reddened `test_link_heal_config`, and `04ef3477a9` reverted it at 07:49. So 6825's solo block at 07:52 as `branch_bug`, and 6161's failure at 07:35, were main red. No `main_health_red` row was emitted for either.

## Rules

- **Tree:** first-parent walk of `merge_sha` to the commit that was main at dispatch. Own members are the top k layers. Each entry ahead is matched to the verify row whose `merge_sha` is that layer.
- **Base state when the verdict returned:**
  - main: nothing unlanded ahead;
  - green: every entry ahead had passed or landed;
  - unverified: some entry ahead had neither;
  - already red.
- **Same content:** same tip, or identical own diff where a task has several tips.
- **Defective, proved:** failed alone with the same content, on main or a base that landed as built, on a test not red on main (8 contents).
- **Defective, inferred:** 7736 at `1b28101f` and 10137, which have no solo.
- **Sound:** same content passed in some tree or landed. "Sound modulo main red" means its only solo failure was a main-red test (10 contents, e.g. 4991). Unknown: 11.
- **Two figures throughout:** "proved" uses only proved labels; "with inferred" adds the two inferred defects and the sound-modulo-main-red contents.
- **Intervals** are Wilson 95% and treat slots as independent. They are not: 7736, 8826 and 5281 recur.

| Base state at verdict | Green | Red |
|---|---|---|
| Main | 74 | 62 (30 while main was red) |
| Green speculative base | 25 | 13 |
| Unverified | 15 | 25 |
| Already red | 0 | 0 |

## 1. Red on an unverified base (n = 25: 16 chains, 9 solos)

| How the base resolved | n | Batches with a defective own member | Per-member defect share |
|---|---|---|---|
| Later green, landed as built | 4 | 3 of 3 proved; 4 of 4 with inferred | 3/18; 4/19 with inferred |
| Later red, main not red | 13 | 3 of 11 proved; 3 of 12 with inferred | 3/51 = 0.06; 4/53 = 0.08 |
| Later red, and main red | 5 | 1 of 4; 1 of 5 with inferred | 1/4; 1/5 |
| Never landed as built, no red verdict | 3 | 2 yes, 1 undetermined | 2/3 (thin) |

- The four "later green" chains are 612323, 612686, 619263 and 625930.
- A red on an unverified base resolved to a red base 18 of 25 times (0.72) and to a landed base 4 of 25 times; 3 never resolved cleanly.
- A green on an unverified base (n = 15) never saw its base go red: 10 landed as built, 5 were re-merged.

## 2. Red on a green base, main not red (n = 45: 32 on main, 13 on a base already passed)

- **Chain-level share with a defective own member:** 43 of 43 determinable with inferred, interval [0.92, 1.00]. Proved only: 21 of 21, with 24 undetermined because 7736 has no solo.
- **Undetermined:** 2, the type-error chain 641863 and the timeout solo 656320. Worst case is 43 of 45 = 0.96.
- **Effective sample:** these 45 batches hold only 9 distinct defective contents.

| Batch size | n | Per-member share, proved | Per-member share, with inferred |
|---|---|---|---|
| 1 | 7 | 6/6 | 6/6 |
| 3 | 16 | 7/21 = 0.33 | 17/44 = 0.39 |
| 4 | 2 | 1/5 | 1/5 |
| 6 | 20 | 14/89 = 0.16 | 27/116 = 0.23 |

That is about 1.1 defective members per red batch at size 3 and 1.35 at size 6.

For contrast, red on a red main (n = 30): 24 failed only the main-red test, and 5 of 29 batches had a defective own member (0.17).

## 3. Red on a red base

- **Already red when the verdict returned:** n = 0. The lane cancels successors first: 43 cascade cancels fell within 5 minutes of the base's red row.
- **Base red later:** n = 18.

| Measure | Proved | With inferred |
|---|---|---|
| Per-member defect share, red on later-red base | 4/55 = 0.07 [0.03, 0.17] | 5/58 = 0.09 [0.04, 0.19] |
| Prior per member slot, all 214 verdicts | 43/476 = 0.09 [0.07, 0.12] | 68/547 = 0.12 [0.10, 0.15] |
| Prior per distinct content that was ever an own member | 8/199 = 0.04 [0.02, 0.08] | 10/211 = 0.05 [0.03, 0.09] |
| Batches with a defective own member, red on later-red base | 4/15 = 0.27 | 4/17 = 0.24 |
| Same, all batches | 35/171 = 0.20 | 58/209 = 0.28 |

Indistinguishable from the prior at this n. Against the green-base share (0.23 to 0.30 per member) it is clearly lower.

## 4. Green on a red base

**Literal instances: 0.** None of the 114 passes has, in its first-parent tree, an entry whose own verify returned red, before or after. So there is no case of the lane tearing a green one down.

- Of the 40 passes on a speculative base, 31 bases landed as built and 9 had passed or would pass but were re-merged.
- Censoring: 18 successors returned a verdict before their base went red, all red; 43 more were cancelled at a median 34% done. Four of the 18 ran the single chained test command, which stops at the first failing leg and so returns red sooner than green.

**Content-level:** 14 red verifies had all their own members later sitting unchanged in a passing tree.

| Red verify | Own members | Why red | What followed | Class |
|---|---|---|---|---|
| 606153, 607107, 607212 | 10139 | loop-blocking gate, red on main | Passed after 10239 landed | Main red |
| 606742 | 6179 | same | same | Main red |
| 606912 | 5918 | same | same | Main red |
| 606947 | 10161 | same | same | Main red |
| 615733, 615772 | 7736 (fixed tip) | retired-wiring guard, red on main | Passed after direct commit `729259fb` | Main red |
| 615738 | 5464 | same | same | Main red |
| 615764, 615810 | 5758 | same | same | Main red |
| 615797 | 4669 | same | same | Main red |
| 610436 | 6161 | link-heal config test, red on main | Passed 09:03 after the revert | Main red |
| 611453 | 5532, 8015, 6832 | history-leg test from 7736 in the unlanded base | Passed without that base in 612329; landed 11-17 | Inherited base |

No fixer and no false failure of a base.

**Green on a red main:** 2 in window. 607519 is 10239 itself, the heal: an own member repairs main by editing `loop_blocking_allowlist.py` and `checkout_anchor.py`. 607563 is 6577 stacked on it. No pass occurred during the other two episodes; direct commits healed both.

## 5. How long the evidence stays weak

Delay from a red verdict on an unverified base to the base's own verdict (n = 24; one base never got one):

| Base resolved | n | p50 | p90 | Max |
|---|---|---|---|---|
| Any | 24 | 12.5 min | 26.6 min | 43.1 min |
| To red | 18 | 13.8 min | 25.3 min | 40.4 min |
| To green | 6 | 10.0 min | too thin | 43.1 min |

## 6. Joint model check ("red iff it holds a defective unlanded request, else red with probability e")

| Trees | n |
|---|---|
| Green, no defective content | 114 |
| Green holding defective content (contradiction) | 0 |
| Red holding proved-defective content | 39 |
| Red holding only inferred-defective content | 29 |
| Red, main-red test only | 28 |
| Red, every unlanded request proved sound, main not red (contradiction) | 0 |
| Red, undetermined | 4 |

- **Undetermined reds:** 641863 (type error), 643973 (history-leg test; 10452 or 5429), 656320 (timeout), 607004 (5285, red from main anyway).
- **e:** 0 of 114, at most 3 of 117 (2.6%). This replaces my earlier 2 of 116. It is after the flake gate; 6 of the 114 passes were flake-suppressed.
- **Repair rate:** 0 of 68 trees holding a defective content were green, and 0 of 10 defective contents ever passed unchanged.
- **Test level:** the defect's own test failed in 81 of 83 tree-content pairs. One was masked by an earlier failing leg. The other is 10242 in 622060, which fails only when 6514 is also present, so "defective" is not always a property of one request.
- **Circularity:** the 29 "inferred" trees are mostly 7736, whose label came from this same pattern plus its fix commit.
- **Stray failures:** five extra tests failed once each, always in trees already red. One (5285's `seed_scratch_reference`) passed in a superset tree on the other host the same minute, so I moved 5285 to unknown.

## Hand-checked from raw rows

- **612686:** 4643's `speculative_merge.base_sha` `60b07056` equals the `merge_sha` of 612704, which passed 4.8 minutes after the red.
- **637195:** red at 08:52 and its base 637929 red at 09:32. Then 6960, 9636, 6979 and 6984 each passed alone and landed on unchanged tips.

## Could not determine

- **604241:** 5189 passed alone on `e8b3ab52` at 10:58, 21 minutes before the window, on a main already red; the same tip failed at 12:35. Pass logs are not archived.
- **7736 at `1b28101f` and 10137:** no solo exists for either.
- **The 43 cancelled successors:** their verdicts were never produced.

Scripts and outputs are in `/tmp/gate-10069/forensics/`: `cond_build.py`, `cond_report.py`, `cond_456.py`, `sig.py`, `hand2.py`, `cond.json`, `cond_report.out`, `cond_456.out`, `sig.out`.
