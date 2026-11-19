# Weekly caps on the Claude subscription accounts: one budget or two?

Evidence seat, read-only. Written 2026-11-15. Scripts, extracts and raw outputs are in
`/tmp/team-weights-1112/followup-1115/caps/`. Every dollar figure is **notional API spend at list price**
(what the API would have charged), unless a line says otherwise.

## Verdict

**The two-budget structure is supported. The sizes in the hypothesis are not.**

What holds:
- There are two weekly limits per account. Response headers name them: `7d` (general) and `7d_oi`
  (Fable only; the claim's name is `seven_day_overage_included`).
- A Fable request needs room under both. On 2026-11-04, Fable was refused on six accounts whose general
  limit was full even though their Fable limit had room. On account A it was refused because the Fable
  limit was full while the general limit stood at 62%.
- Non-Fable models keep working after the Fable limit fills.

What does not hold, or differs:
1. **Fable counts against the general limit, at about twice its list price.** Each $1 of Fable 5.1
   used about $2 of the general budget. So the general limit covers all models, not "all except Fable".
2. **Non-Fable spend does not count against the Fable limit.** Accounts that had spent about $2,000 on
   Opus and Sonnet showed the Fable limit at 0–1%.
3. **The Fable limit is not half the general limit.**
   - It is about **$435 of Fable 5.1 per week** at list price, about one fifth of the general limit.
   - Measured in general-limit units (Fable counted double) it is about $870, which is **0.40 of the
     general limit** (range 0.36–0.54).
   - "Half" sits at the edge of that range. It holds only if Fable counts about 2.5 times against the
     general limit. The one direct reading gives 2.0 times, so it points away from half.

**Best alternative reading:**
- General weekly limit C ≈ **$2,200 per account**, counting all models at list price with Fable 5.1
  at ×2.
- Separate Fable-only weekly limit F ≈ **$435 of Fable 5.1** at list price, counting only Fable.
- A Fable request is served only while both limits have room. Every other model needs only the general
  limit.

## The two cap figures

| figure | value | rule | window | population | kind |
|---|---|---|---|---|---|
| General weekly limit C (fleet) | **$2,188** median, range $2,062–2,268 | Recorded spend from the weekly reset to the first general weekly cap. Factory + proxy-matched interactive + fingerprinted interactive + reconciliation harness. List price, Fable 5.1 ×2, Fable 5.2 ×1. | windows starting 2026-11-01 11:00 to 11-12 | max-d…max-h, 10 cap events, each cap pinned to within 1 h by the last service before it | **measured lower bound** (evals, census and claude.ai use are not in it) |
| same, including max-b and max-c | $2,169 median, range $1,941–2,268 | same | same | 12 cap events | lower bound |
| same, no Fable weighting | $2,116 median, range $1,486–2,244 | same, all models at list price | same | 12 cap events | lower bound |
| General weekly limit, account A | $1,698 and $1,832; at most $1,903 / $1,969 if all unattributed interactive spend was A's | same rule | A's windows starting 11-02 and 11-09 | 2 cap events | lower bound |
| Fable-only weekly limit F | **$435** central, range $412–454 | Fable 5.1 list-price spend from the reset to the Fable-only cap (A $431, max-c $454). Or Fable spend ÷ `7d_oi` utilisation (max-g 72/0.16 = $447, max-h 78/0.19 = $412). | 2026-11-02 to 11-06 | 4 measurements: 2 cap events on A and max-c, 2 header readings on max-g and max-h | measured; header ratios exact to ±3% (rounding) |
| Fable 5.1 weight in the general limit | **2.0×** (1.97–2.08 from the reading's rounding; 1.85–2.69 if A's unattributed spend is placed at either extreme) | Solve (non-Fable + w·Fable) at the reading ÷ the same at the cap = 0.62, using A's 11-04 08:27 reading. A sold no Fable between the reading and the cap. | A, window 11-02 | 1 reading + 1 cap | measured; independent check: across the 14 clean windows the at-cap spread is lowest for w between 2.0 and 3.0 |
| F / C | 0.40 (0.36–0.54) | 435 × w ÷ 2,188 | as above | as above | derived |

## Direct readings of quota use (the strongest evidence)

There is no running series of utilisation readings anywhere:
- The runs.db tables, the proxy log, the `.cds` files and a sample of about 45 archived agent transcripts
  hold no utilisation values.
- Those transcripts record a quota only when a request is refused (`quotaLimits`, `rate_limit_info`:
  status, reset time, type), never as a fraction.
- The proxy writes only its cooldown deadline to the `.cds` files and to its log.

Two interactive sessions quoted response headers word for word, though:

| time (UTC) | account | `7d` used | `7d_oi` used | source |
|---|---|---|---|---|
| 11-01 12:58 | max-b | 0.45 | 0.10 | `~/.claude/projects/-home-leo-src-dark-factory/3aa12a76-6caf-46fe-b586-8e421246a8e5.jsonl` (proxy management snapshot) |
| 11-04 08:27 | max-a | 0.62 | **1.01, rejected** (claim `seven_day_overage_included`) | `bb6e548e-566a-4c49-9193-a0b46e7696ab.jsonl` (one small Fable request per account) |
| 11-04 08:27 | max-b | 1.0, rejected | 0.99 | same |
| 11-04 08:27 | max-c | 1.0, rejected | 0.38 | same |
| 11-04 08:27 | max-d / max-e / max-f | 1.0, rejected | 0.01 / 0.01 / 0.00 | same |
| 11-04 08:27 | max-g / max-h | 1.0, rejected | 0.16 / 0.19 | same |

Matched against recorded spend in the same windows (`utilization_out.txt`):

**Non-Fable spend does not count in `7d_oi`.**
- max-d, max-e and max-f had about $2,000 of non-Fable spend each and almost no Fable ($3, $6, $23).
- Their `7d_oi` readings were 0.00–0.01.

**`7d_oi` fills at about $435 of Fable 5.1.**
- max-g: $72 of recorded Fable → 0.16.
- max-h: $78 → 0.19.
- A: $431 → rejected. A's last Fable request was served at 11-03 17:54, and the first Fable refusal
  came in the same minute.

**Fable counts about double in `7d`.**
- At the 11-04 reading, A had spent $187 non-Fable and $431 Fable 5.1, and stood at 0.62.
- From the reading to its general cap at 11-06 00:03, A spent only non-Fable: $649 more, to $836.
- Counting Fable once gives a ratio of 0.49 at the reading, not 0.62. A limit that ignored Fable would
  give 0.22. Counting Fable twice gives 0.62 exactly.

**There is unrecorded use on max-b and max-c.**
- max-c showed `7d_oi` at 0.38 with $0 of recorded Fable in its window. That is about $165 of Fable
  that no store holds.
- max-b showed 0.99 against $307 recorded, so about $130 is missing.
- max-b also went from `7d` 0.45 at 12:58 to capped by 19:04 on 11-01. Only about $530 of
  general-limit spend was recorded in that time, so at least $400 more went unrecorded.

There is only one usable mid-window `7d` reading below 1.0 (A). max-b's 11-01 reading cannot be used:
max-b's cap moment is uncertain by 6 hours, and part of its spend was not recorded. So the weight
"Fable 5.1 = 2×" rests on one reading, plus the at-cap spread across windows (next section).

## Predictions, tested

| # | prediction (the hypothesis as stated) | verdict | evidence |
|---|---|---|---|
| 1 | Spend at a general weekly cap is one constant C across accounts and weeks | **Holds for max-b…max-h** once Fable 5.1 counts ×2. **Fails for account A**, which caps 15–20% lower. | 10 clean max-d…max-h windows: $2,062–2,268, spread (coefficient of variation) 2.5%. Every max-d…max-h window since 09-10: 9.5%, with every low week explainable (prediction 3). max-b and max-c clean weeks: $1,941–1,999. A: $1,698 and $1,832. |
| 2a | Fable spend at a Fable-only cap is one constant | **Holds** (4 measurements, $412–454) | table above |
| 2b | …and it equals C/2 | **Not supported.** $435 is 0.20 of C at list price, or 0.40 of C counting Fable ×2 (range 0.36–0.54). | F and the Fable weight above |
| 3 | A general cap never arrives with spend well below C unless unrecorded consumers explain it | **Holds for max-b…max-h; not settled for A.** | Every low pre-proxy week has more unattributed interactive spend in its window than its shortfall. For max-b 10-30 and max-c 10-28 the headers prove unrecorded use on those accounts (above). Since the proxy (11-01), only A's two weeks are low. |
| 4 | Fable is never served after the Fable cap and before the reset | **Holds** (2 of 2 events) | A: 0 Fable served from 11-03 17:54 to the 11-09 reset, 46 Fable refusals. max-c: 0 Fable served from 11-04 21:36 to the 11-11 reset, 19 refusals (`checks_out.txt`). |
| 5 | Non-Fable models are served after a Fable cap | **Holds** (2 of 2) | A: 2,837 non-Fable requests served after its Fable cap, until its general cap. max-c: 1,882. |
| 6 | Fable needs room under both caps | **Holds** | 11-04 08:27: six accounts refused Fable with claim `seven_day` when `7d` = 1.0, although `7d_oi` was 0.00–0.99. A refused Fable with claim `seven_day_overage_included` when `7d_oi` = 1.01 and `7d` = 0.62. |
| 7 | (added) Nothing is served after a general weekly cap until the reset | **Holds, with one anomaly** | 5 cases of 0.1–6 h are single requests already in flight ($0–8). The exception: max-d capped 09-27 11:01 (reset 10-01 13:00), then a probe succeeded at 09-28 07:33 and it served $1,621 before the reset. That is an early lift, cause unknown. |

**Does Fable count against the general budget?** Yes, at about twice its list price. Without Fable:
- A would have capped at $836 and max-c at $1,032, against about $2,200 for everyone else.
- The A reading would be off by a factor of 2.8.
So "C covers all models except Fable" is refuted.

## Which measure the limit counts

Population: the 14 clean windows (they start after the proxy began logging accounts, 11-01 10:50, and
the cap is pinned to within 1 hour). The same comparison over all 46 pinned windows since 09-10 ranks
the measures the same way (`measures_out.txt`, `ratio_test_out.txt`, `fit_out.txt`).

| measure | spread at cap (CV), 14 windows | without A (12) | ratio at A's 0.62 reading |
|---|---|---|---|
| recorded cost_usd, all models at list | 0.141 | 0.096 | 0.49 |
| non-Fable cost only | 0.225 | 0.164 | 0.22 |
| **list cost, Fable 5.1 ×2, Fable 5.2 ×1** | **0.078** | **0.045** | **0.62** |
| list cost, all Fable ×2 | 0.097 | 0.076 | 0.62 |
| list cost without cache reads | 0.105 | – | 0.50 |
| all tokens (input + output + cache read + cache write) | 0.245 | 0.170 | 0.45 |
| tokens without cache reads | 0.083 | 0.089 | 0.35 |
| output tokens only | 0.307 | 0.201 | 0.67 |

**`cost_usd` already is list price.** A per-row check gives a median ratio of 1.0000 and a 10–90%
range of 1.0000 to 1.0000 (`pricefit_out.txt`):

| model | input | output | cache read | cache write |
|---|---|---|---|---|
| Opus 5.5 | $4 | $20 | $0.20 | $8 (the 1-hour rate) |
| Sonnet 5.5 | $2 | $10 | $0.10 | $4 |
| Fable 5.1 | $10 | $50 | $0.25 | $20 |

Prices are per million tokens. Cache writes are charged at the 1-hour rate, twice the input price. Rows
from before 10-07 carry no model id; they match Opus 5 or Sonnet 5 prices or the 5.5 ones, by date
(`generations.py`). So "recomputed from tokens" and "recorded" are the same number.

**Dollars fit better than tokens.** Re-pricing every Opus and Sonnet token at one generation's prices
(which is what a fixed token budget would mean) widens the spread across the September→October Opus 5 →
Opus 5.5 price change. CV rises from 0.095 to 0.146 over 32 max-d…max-h windows.

**A free fit of separate weights adds little.** It halves output weight and moves cache weights. That
gets the spread to 0.047, which with 5 weights and 14 windows is not better than the simple rule.

**Conclusion:** the limit counts list-price dollars, with Fable 5.1 at about 2×.

**Fable 5.2 looks lighter.**
- One window decides it: max-c from 11-11, with $493 of Fable 5.2 and $0 of 5.1.
- That window fits the general limit with 5.2 at ×1 ($2,136). At ×2 it would be $2,629, an outlier.
- It also spent more than $435 of Fable 5.2 without a Fable-only cap. So 5.2 also counts less than 5.1
  per dollar in `7d_oi`, or not at all.
- Treat this as one window, not a finding.

## Per account, per week: spend from the reset to the first general cap

Columns:
- **Fable $, non-Fable $, total $**: list price.
- **total, Fable 5.1 ×2**: general-limit units.
- **interactive / recon**: the part of the total from interactive sessions (proxy-matched or session
  fingerprint) and from the reconciliation harness.
- **unknown-account interactive**: interactive spend in the same window that cannot be tied to any
  account. It is an upper bound on what is missing, shared across all accounts, not per account.

Marks:
- **LOW**: under 85% of C = $2,200 at a cap.
- **explainable**: the unknown pool is at least the shortfall.
- **UNEXPLAINED**: it is not.
- **unpinned**: last service more than 1 hour before the detection. The spend shown runs to the detection.
- **partial**: the window began before the extract (09-10) or is still open.

Rule: one cap per (account, kind, reset). Its time is the earliest of:
- a factory `cap_hit` whose message is a real CLI banner;
- an interactive weekly-limit refusal whose reset falls at that account's reset hour;
- a proxy upstream refusal (429) on a non-Fable model.

Weekly resets are each account's fixed hour (UTC): max-c Wed 10:00, max-f Wed 13:00, max-d Thu 13:00,
max-b Fri 12:00, max-e Sat 17:00, max-g Sun 13:00, max-h Mon 21:00, max-a Mon 13:00. A burst of 6
`cap_hit` rows on 09-20 01:16–01:23 quoted banner text from agent output ("resets Aug 5"). Those rows
were excluded. The 11-12 study's window check had already dropped them.

| account | window start (UTC) | first general cap | cap pinned within | Fable $ | non-Fable $ | total $ (list) | total, Fable 5.1 x2 | of which interactive / recon | unknown-account interactive in window | mark |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| max-a | 10-19T13:00 | 10-25T15:38 | - | 7 | 138 | 145 | 153 | 145 / 0 | 3,711 | unpinned; LOW; explainable (before the proxy, A's own use is unattributed) |
| max-a | 10-26T13:00 | 11-02T11:04 | 0.0 h | 194 | 435 | 629 | 824 | 629 / 0 | 2,260 | LOW; explainable (same) |
| max-a | 11-02T13:00 | 11-06T00:03 | 0.0 h | 431 | 836 | 1,267 | 1,698 | 1,267 / 0 | 205 | LOW; UNEXPLAINED; Fable-only cap 11-03T17:54 |
| max-a | 11-09T13:00 | 11-11T17:12 | 0.7 h | 174 | 1,609 | 1,783 | 1,832 | 1,783 / 0 | 137 | LOW; UNEXPLAINED |
| max-b | 09-11T12:00 | 09-13T09:56 | 0.0 h | 10 | 1,625 | 1,635 | 1,645 | 2 / 0 | 1,241 | LOW; explainable |
| max-b | 09-18T12:00 | 09-20T19:15 | 0.0 h | 28 | 2,535 | 2,563 | 2,590 | 2 / 0 | 1,133 |  |
| max-b | 09-25T12:00 | - | - | 2 | 229 | 231 | 234 | 0 / 0 | 3,247 | no general cap |
| max-b | 10-09T12:00 | 10-13T15:06 | 0.1 h | 0 | 1,129 | 1,129 | 1,129 | 2 / 0 | 2,080 | LOW; explainable |
| max-b | 10-16T12:00 | 10-22T09:24 | 0.1 h | 5 | 2,291 | 2,296 | 2,301 | 2 / 0 | 4,132 |  |
| max-b | 10-23T12:00 | 10-26T07:55 | 0.0 h | 0 | 1,937 | 1,937 | 1,937 | 1 / 34 | 876 |  |
| max-b | 10-30T12:00 | 11-01T19:04 | 6.0 h | 265 | 769 | 1,034 | 1,299 | 349 / 19 | 1,079 | unpinned; LOW; explainable, and headers prove unrecorded use |
| max-b | 11-06T12:00 | 11-07T10:05 | 0.0 h | 90 | 1,819 | 1,909 | 1,999 | 373 / 32 | 81 |  |
| max-c | 09-09T10:00 | 09-11T03:20 | 0.0 h | 1 | 1,183 | 1,184 | 1,185 | 12 / 0 | 1,094 | partial |
| max-c | 09-16T10:00 | 09-20T20:54 | 0.1 h | 20 | 2,370 | 2,390 | 2,410 | 1 / 0 | 2,301 |  |
| max-c | 09-23T10:00 | 09-25T05:00 | 0.0 h | 17 | 1,964 | 1,981 | 1,998 | 1 / 0 | 927 |  |
| max-c | 09-30T10:00 | - | - | 0 | 295 | 295 | 295 | 0 / 0 | 2,604 | no general cap |
| max-c | 10-07T10:00 | 10-11T09:00 | 5.9 h | 36 | 1,073 | 1,109 | 1,144 | 311 / 0 | 2,571 | unpinned; LOW; explainable (a terminal session was fingerprinted on max-c on 10-11) |
| max-c | 10-14T10:00 | - | - | 1 | 292 | 293 | 295 | 0 / 0 | 5,909 | no general cap |
| max-c | 10-21T10:00 | 10-24T12:53 | 0.0 h | 7 | 2,628 | 2,635 | 2,642 | 6 / 0 | 1,820 |  |
| max-c | 10-28T10:00 | 11-01T05:05 | 19.7 h | 0 | 1,139 | 1,139 | 1,139 | 404 / 6 | 1,253 | unpinned; LOW; explainable, and headers prove unrecorded Fable use |
| max-c | 11-04T10:00 | 11-06T17:30 | 0.0 h | 454 | 1,032 | 1,486 | 1,941 | 733 / 9 | 140 | Fable-only cap 11-04T21:36; low at list price only |
| max-c | 11-11T10:00 | 11-14T16:06 | 0.0 h | 493 | 1,643 | 2,136 | 2,136 | 1,809 / 26 | 133 | all Fable here is 5.2 (counted ×1) |
| max-d | 09-10T13:00 | 09-13T17:33 | 1.1 h | 0 | 1,450 | 1,450 | 1,450 | 173 / 0 | 2,532 | unpinned; LOW; explainable |
| max-d | 09-17T13:00 | 09-22T19:18 | 0.0 h | 24 | 2,483 | 2,507 | 2,531 | 0 / 0 | 3,348 |  |
| max-d | 09-24T13:00 | 09-27T11:01 | 0.1 h | 13 | 1,905 | 1,919 | 1,932 | 0 / 0 | 922 | early lift: served $1,621 from 09-28 07:33, before the reset |
| max-d | 10-01T13:00 | 10-03T04:36 | 0.0 h | 9 | 2,130 | 2,139 | 2,148 | 1 / 0 | 625 |  |
| max-d | 10-08T13:00 | 10-10T04:27 | 0.0 h | 0 | 1,994 | 1,994 | 1,994 | 1 / 0 | 1,571 |  |
| max-d | 10-15T13:00 | 10-17T05:35 | 0.0 h | 0 | 1,557 | 1,557 | 1,557 | 1 / 13 | 1,594 | LOW; explainable |
| max-d | 10-22T13:00 | 10-23T23:30 | 0.0 h | 2 | 1,642 | 1,644 | 1,646 | 4 / 48 | 913 | LOW; explainable |
| max-d | 10-29T13:00 | 10-31T01:09 | 0.0 h | 3 | 2,090 | 2,093 | 2,096 | 1 / 66 | 721 |  |
| max-d | 11-05T13:00 | 11-06T17:06 | 0.1 h | 58 | 2,077 | 2,135 | 2,192 | 58 / 52 | 88 |  |
| max-d | 11-12T13:00 | 11-14T09:43 | 0.0 h | 24 | 2,219 | 2,244 | 2,268 | 15 / 185 | 70 |  |
| max-e | 09-05T17:00 | - | - | 0 | 181 | 181 | 181 | 0 / 0 | 3,147 | partial; no general cap |
| max-e | 09-12T17:00 | 09-15T09:58 | 0.0 h | 13 | 1,635 | 1,648 | 1,661 | 2 / 0 | 910 | LOW; explainable |
| max-e | 09-19T17:00 | - | - | 7 | 1,092 | 1,099 | 1,106 | 0 / 0 | 3,803 | no general cap |
| max-e | 09-26T17:00 | 10-01T03:36 | 0.0 h | 5 | 1,809 | 1,814 | 1,819 | 1 / 0 | 2,206 | LOW; explainable |
| max-e | 10-03T17:00 | 10-05T08:17 | 0.0 h | 2 | 2,092 | 2,094 | 2,096 | 1 / 0 | 462 |  |
| max-e | 10-10T17:00 | 10-12T06:08 | 0.0 h | 6 | 1,972 | 1,978 | 1,984 | 0 / 0 | 466 |  |
| max-e | 10-17T17:00 | 10-19T15:47 | 0.1 h | 6 | 2,140 | 2,146 | 2,152 | 0 / 10 | 1,436 |  |
| max-e | 10-24T17:00 | 10-26T02:14 | 0.0 h | 0 | 1,972 | 1,972 | 1,972 | 2 / 59 | 483 |  |
| max-e | 10-31T17:00 | 11-02T00:35 | 0.0 h | 6 | 2,171 | 2,177 | 2,183 | 0 / 57 | 630 |  |
| max-e | 11-07T17:00 | 11-08T10:14 | 0.0 h | 13 | 2,036 | 2,049 | 2,062 | 334 / 38 | 28 |  |
| max-e | 11-14T17:00 | - | - | 36 | 1,004 | 1,041 | 1,039 | 160 / 31 | 23 | partial (open); no general cap |
| max-f | 09-09T13:00 | 09-16T09:55 | 0.0 h | 10 | 1,286 | 1,296 | 1,306 | 2 / 0 | 4,366 | partial |
| max-f | 09-16T13:00 | - | - | 0 | 85 | 85 | 85 | 0 / 0 | 4,036 | no general cap |
| max-f | 09-23T13:00 | - | - | 15 | 748 | 763 | 778 | 46 / 0 | 3,299 | no general cap |
| max-f | 09-30T13:00 | 10-05T10:29 | 0.0 h | 1 | 2,215 | 2,216 | 2,217 | 0 / 0 | 1,553 |  |
| max-f | 10-07T13:00 | 10-10T20:48 | 2.4 h | 4 | 2,055 | 2,059 | 2,064 | 0 / 0 | 2,365 | unpinned |
| max-f | 10-14T13:00 | 10-17T13:01 | 0.1 h | 24 | 1,986 | 2,009 | 2,033 | 221 / 11 | 2,760 |  |
| max-f | 10-21T13:00 | 10-23T00:18 | 0.0 h | 0 | 1,979 | 1,979 | 1,979 | 87 / 0 | 1,224 |  |
| max-f | 10-28T13:00 | 10-31T05:08 | 0.1 h | 2 | 1,880 | 1,882 | 1,884 | 2 / 58 | 903 |  |
| max-f | 11-04T13:00 | 11-05T11:53 | 0.1 h | 148 | 1,870 | 2,019 | 2,167 | 151 / 47 | 43 |  |
| max-f | 11-11T13:00 | 11-14T21:33 | 0.0 h | 79 | 2,159 | 2,238 | 2,251 | 197 / 124 | 143 |  |
| max-g | 09-13T13:00 | - | - | 4 | 347 | 351 | 356 | 0 / 0 | 3,444 | no general cap |
| max-g | 09-27T13:00 | - | - | 6 | 1,039 | 1,045 | 1,051 | 0 / 0 | 3,002 | no general cap |
| max-g | 10-04T13:00 | 10-07T11:01 | 0.0 h | 17 | 2,242 | 2,259 | 2,276 | 2 / 0 | 1,120 |  |
| max-g | 10-11T13:00 | 10-13T11:30 | 0.0 h | 0 | 2,167 | 2,167 | 2,167 | 0 / 0 | 369 |  |
| max-g | 10-18T13:00 | 10-21T05:08 | 0.0 h | 9 | 2,030 | 2,039 | 2,049 | 1 / 1 | 1,972 |  |
| max-g | 10-25T13:00 | 10-27T04:01 | 0.0 h | 3 | 2,017 | 2,019 | 2,022 | 2 / 54 | 551 |  |
| max-g | 11-01T13:00 | 11-02T22:21 | 0.0 h | 72 | 2,016 | 2,088 | 2,159 | 81 / 54 | 418 |  |
| max-g | 11-08T13:00 | 11-09T16:00 | 0.0 h | 56 | 2,059 | 2,114 | 2,170 | 231 / 54 | 64 |  |
| max-h | 09-07T21:00 | - | - | 0 | 87 | 87 | 87 | 0 / 0 | 3,690 | partial; no general cap |
| max-h | 09-21T21:00 | 09-27T18:40 | 289.6 h | 0 | 43 | 43 | 43 | 43 / 0 | 2,485 | not a real week: max-h was barely in use; one terminal session hit its limit |
| max-h | 09-28T21:00 | - | - | 0 | 107 | 107 | 107 | 107 / 0 | 3,127 | no general cap |
| max-h | 10-12T21:00 | - | - | 0 | 1,328 | 1,328 | 1,328 | 0 / 0 | 5,810 | no general cap |
| max-h | 10-19T21:00 | - | - | 0 | 1,035 | 1,035 | 1,035 | 8 / 0 | 4,003 | no general cap |
| max-h | 10-26T21:00 | 10-28T05:51 | 0.0 h | 4 | 1,965 | 1,969 | 1,973 | 5 / 46 | 204 |  |
| max-h | 11-02T21:00 | 11-03T19:57 | 0.0 h | 78 | 2,039 | 2,117 | 2,194 | 94 / 133 | 85 |  |
| max-h | 11-09T21:00 | 11-10T22:37 | 0.0 h | 3 | 2,238 | 2,241 | 2,244 | 2 / 56 | 117 |  |

Machine-readable copy: `caps/table.csv`.

## The low weeks on max-b and max-c

The 11-12 study's "about a third" weeks were max-b 10-09 and 10-30, and max-c 10-07, 10-28 and 11-04.
They come out like this:

- **max-c 11-04 is not low.**
  - At list price it is $1,486, but $454 of it is Fable 5.1.
  - Counting Fable ×2, as the header reading says the limit does, gives $1,941, inside the clean range.
  - The same applies to A's 11-02 week: $1,267 at list price, $1,698 with Fable ×2.

- **max-b 10-30 and max-c 10-28 had consumption that no store recorded.** The 11-04 headers prove it:
  - max-c's Fable limit stood at 0.38 with $0 of recorded Fable.
  - max-b's stood at 0.99 against $307 recorded.
  - max-b's general limit went from 0.45 to full in 6 hours, with at most $530 recorded in that time.

- **All pre-proxy low weeks fit inside the unattributed interactive spend.**
  - Before 11-01, transcripts on no identifiable account ran **$2,800–5,500 a week**, almost all from
    terminal (`cli`) sessions (`unknown_pool_out.txt`).
  - That is $1,000–3,500 a week more than account A could have absorbed at its own cap (at most about
    $1,970).
  - So the excess must have run on fleet tokens.
  - Session fingerprints confirm terminal sessions on fleet accounts: max-d on 09-13, max-h on 09-27,
    max-c on 10-11, max-f on 10-17 and 10-19, and max-b on 11-01.
  - The weekly shortfalls ($550–1,070 per low week) are well within this.
  - What cannot be shown is which session ran on which account in any one of those weeks.

- **Since the proxy began logging accounts (11-01), max-b and max-c cap at $1,941–1,999.** That is
  10% under max-d…max-h, within the spread a little unrecorded use would cause. They are no longer bimodal.

**How much interactive overflow each account took per week** (proxy era, the proxy log joined to
transcripts, list price). The largest was max-c's 11-11 week at $1,809:

| account (week from) | interactive spend through the proxy and fingerprints |
|---|---|
| max-b (11-06) | $373 |
| max-c (11-04) | $733 |
| max-c (11-11) | $1,809 |
| max-d (11-05) | $58 |
| max-e (11-07) | $334 |
| max-f (11-04) | $151 |
| max-f (11-11) | $197 |
| max-g (11-01) | $81 |
| max-g (11-08) | $231 |
| max-h (11-02) | $94 |

How the matching works:
- 39,415 of 39,980 served proxy requests matched a transcript message on model and time.
- Unmatched messages fall back to a session fingerprint, then to "unknown".
- After 11-01 the unknown pool is $330–410 a week. Only $57–85 of it is from T3 or SDK sessions; the
  rest is scripts (`sdk-cli`).

## Account A as an eighth case

**A's plan is not known to be the same as the fleet's.**
- A's reset is Mon 13:00.
- An 11-04 note says the accounts are on a plan with no 5-hour window. A's headers on 11-04 carried no
  5-hour fields, the same as the others.

**A's general cap comes in lower.**
- A's two clean weeks capped at $1,698 and $1,832 in general-limit units, or at most $1,903 and $1,969
  if every unattributed interactive dollar in those windows was A's.
- That is 10–22% under the fleet's $2,188.

**A's Fable limit matches the fleet's.** A Fable-capped at $431, inside the fleet's $412–454. So F looks
the same on A.

**Two readings, which the data cannot tell apart:**
1. A has some consumption no transcript records, such as claude.ai web, desktop or mobile chat, which
   shares the subscription. But the 0.62 reading leaves no room for such use spread evenly over the week.
2. A's general limit really is about 80–90% of the fleet's.

## Change over time

**The 5-hour window disappeared at the end of October.**
- The last factory session-limit (5-hour) caps were on 10-30 and the last in transcripts on 10-29.
- None since.

**No step in C can be seen at that point.**
- max-d…max-h windows before 10-30: median $2,033 (general-limit units).
- From 10-31 on: median $2,180.
- The 7% rise is the size of the coverage gains in the same weeks, so it cannot be told apart from them:
  - reconciliation harness spend added from 10-17: $10–185 per window;
  - interactive overflow attributed from 11-01.

**Opus 5 → Opus 5.5 (late September) did not move the cap in dollars.**
- Only 2 windows are Opus-5-heavy ($1,661 and $2,531), too few to say more.
- Re-pricing to a fixed token budget fits worse (see "Which measure").

## Consumers checked for coverage

- `curator_events.db` covers the fused-memory **task curator** (roles `task_curator`,
  `task_curator_batch`, `path_scope_adjudicator`).
- It does **not** cover the **reconciliation harness**. I added that from its archived transcripts
  (`agent-transcripts/recon-*`, 16,870 messages): $139, $306, $391 and $453 in ISO weeks 43–46.
  Its account is modelled as the gate's choice: the first account in file order not inside a weekly cap.
  Real choices can differ; session caps and probes are ignored.
- Nightly census and evals: only as far as they write transcripts under `~/.claude/projects`.
  - Script-launched (`sdk-cli`) spend there is $230–620 a week and lands in the unknown pool.
  - Anything run under another `CLAUDE_CONFIG_DIR` is missing.
- claude.ai web, desktop and mobile use is invisible.
- Remote task legs on leo-laptop, if they run agents on these accounts, record nowhere on this host.
  Not checked: that would need access to the laptop.

## What would falsify this

**A clean full-week utilisation series.** A second mid-week `7d` reading on any account, with the same
weighting, would test the core claims:
- If spend over the interval ÷ change in utilisation came out far from $2.2k, the main finding falls.
- If it gave a Fable weight far from 2, the weighting falls.
- If the Fable limit, read the same way, were about $1.1k of Fable 5.1, "not half" falls and the
  hypothesis is restored.

**A fleet week capping clean below ~$1.9k.** That would break "one constant C". Clean means after
11-01, a pinned cap, and little unknown spend.

**Fable served on an account after its `7d_oi` refusal and before the reset.**

**A Fable 5.2-heavy week on a fleet account that Fable-caps near $435 of 5.2.** That would show 5.2
counts like 5.1. Today's single max-c week says it counts less.

## What could not be determined

- **The Fable 5.1 weight in `7d`.** It rests on one reading (2.0×). The at-cap spread allows 2.0–3.0,
  and so the F/C ratio is only bounded to 0.36–0.54.
- **How Fable 5.2 counts in either limit.** One week.
- **Why A caps lower**: unrecorded use (claude.ai) or a smaller plan.
- **Which accounts the pre-proxy terminal sessions used week by week.** The low weeks are explainable
  but not attributed.
- **max-d's early lift on 09-28** (served $1,621 before its reset).
- **Units of `7d_oi`.** Only its fill point in Fable dollars is known, not whether it is "half" of
  anything internally.
- The proxy keeps the last response's `anthropic-ratelimit-unified-*` headers in memory. It exposes
  them only through its management API, which needs the management key. I did not read it.

## Proposals (not run)

1. **Free and local (no Anthropic request).**
   - Poll the proxy's management endpoint (`GET /v8/management/credentials`, field `quota.signals`)
     every few minutes and log `7d` and `7d_oi` utilisation, with a timestamp, for every proxied account.
   - Regressing cumulative recorded spend since the reset against `7d` gives C. Fable spend against
     `7d_oi` gives F. The joint fit gives the Fable weight.
   - A week of this would settle every open item above.
   - It needs the management key, so it is Leo's call.
2. **One direct probe per account.**
   - One `max_tokens=1` Haiku request per setup token at two times about a day apart. Each costs a few
     tokens; a refused request costs nothing.
   - Read `anthropic-ratelimit-unified-7d-utilization` and `-7d_oi-utilization` each time.
   - With recorded spend between the two times, this gives C per account and A's plan directly.
   - This sends requests to Anthropic, so it was not run.

## Files

`caps/`:
- `extract.py` → `raw.pkl`: fresh extract of the 10 stores, to 11-15 09:36.
- `pricefit.py`: cost_usd = list price.
- `interactive_extract.py` → `interactive.pkl`: 322k messages since 09-10.
- `proxy_parse.py` → `proxy.pkl`.
- `join.py` → `attributed.pkl`.
- `recon_extract.py` → `recon.pkl`.
- `analysis.py` → `analysis_out.txt`: per window.
- `utilization.py` → `utilization_out.txt`.
- `measures.py`, `ratio_test.py`, `fit.py`, `checks.py`, `generations.py`: their outputs are `*_out.txt`.
- `table.py` → `table.md` and `table.csv`.
