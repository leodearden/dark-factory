# Gate 10069, sixth run: fallback for a red batch that cannot be scanned

**Verdict.** Do not ship the existing inference fallback (F3) as it stands. When "cannot be scanned" is a property of the culprit (sticky), F3 keeps re-batching the same set and the lane loses 82–96% at m ≥ 0.10 (63–86% at m=0.05). Halving (F2) is not worth its state: under sticky it buys +14/+11/+4/+2% over each-alone (F1) at q = 0.05/0.08/0.14/0.20, and is beaten at m ≥ 0.10 by a simpler variant I added, F1u (each-alone, and the rest of the batch returns as fresh once one member blocks). Whatever the fallback, m=0.16 sticky costs 26–61% of the corrected lane, so lowering m is worth more than the choice of fallback.

**How figures are counted.** Capacity-bound (75 waiting), e=0.01, main-red and restarts on, always stacked, today's cascade, caps 12/24, 120 h window, n=60, same seeds per cell. "lvh" is landings per verify-hour; Δ% is paired, ±95% CI. The real lane sees about 2.4 arrivals/h.

**Reproduction before any change.** q=0.14, caps 12/24, default scan: U 1.40, U+S 4.24 (+204 ±4), miss 0.20 −13 ±1. These match `results2_S_compact.txt`, and I re-ran the comparison after each of the eleven engine patches.

**Design changes applied.**
- The scan is member-alone: n+2 trees, plus one confirm tree per culprit (T1).
- An unscannable red batch spends 2·c_t before its fallback.
- T2 is reported once and is off in the comparison.
- Sticky: a red is unscannable if any culprit in its tree is.
- "Corrected lane" below means n+2 trees, T1, T3 with one scan resource, T4.

**Two variants are mine, not in the brief.**
- **F1u:** as F1, but each alone member remembers its batch; when one member blocks, the rest return as fresh.
- **F3b:** as F3, but a batch holding a member of a red observation is priced with plain U's going cost instead of the scan's.

## (a) Fallbacks × q × m

Each cell is lvh / landings per hour (Δ% vs corrected lane at m=0), then the batch size k the rule picks at the prior.

**Sticky (the realistic model), W1**

| q | m | F1 alone | F1u | F2 halve | F3 as is | F3b | F4 bisect |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.05 | 6.64 / 13.2 (−21 ±3) k12 | 7.30 / 14.5 (−13 ±2) k12 | 7.07 / 14.1 (−15 ±2) k12 | 3.12 / 6.2 (−63 ±7) k12 | 7.43 / 14.8 (−11 ±2) k12 | 7.30 / 14.5 (−13 ±2) k12 |
| 0.05 | 0.10 | 5.57 / 11.1 (−33 ±3) k12 | 6.39 / 12.7 (−24 ±3) k12 | 6.12 / 12.2 (−27 ±2) k12 | 1.65 / 3.3 (−80 ±5) k12 | 6.78 / 13.5 (−19 ±2) k12 | 6.40 / 12.7 (−23 ±3) k12 |
| 0.05 | 0.16 | 4.79 / 9.5 (−43 ±3) k10 | 5.73 / 11.4 (−31 ±3) k12 | 5.46 / 10.9 (−35 ±3) k12 | 1.01 / 2.0 (−88 ±4) k12 | 6.20 / 12.3 (−26 ±2) k12 | 5.82 / 11.6 (−30 ±3) k12 |
| 0.05 | 0.25 | 4.07 / 8.1 (−51 ±2) k9 | 4.84 / 9.7 (−42 ±3) k12 | 4.66 / 9.3 (−44 ±3) k12 | 0.81 / 1.6 (−90 ±4) k12 | 5.38 / 10.7 (−36 ±2) k12 | 5.00 / 10.0 (−40 ±3) k12 |
| 0.08 | 0.05 | 5.35 / 10.6 (−24 ±3) k12 | 5.78 / 11.5 (−18 ±3) k12 | 5.83 / 11.6 (−17 ±3) k12 | 1.84 / 3.7 (−74 ±6) k12 | 6.06 / 12.0 (−14 ±2) k12 | 5.87 / 11.7 (−17 ±3) k12 |
| 0.08 | 0.10 | 4.29 / 8.5 (−39 ±3) k10 | 4.87 / 9.7 (−31 ±2) k12 | 4.73 / 9.4 (−33 ±3) k12 | 0.96 / 1.9 (−86 ±4) k12 | 5.38 / 10.7 (−24 ±2) k12 | 4.89 / 9.7 (−31 ±2) k12 |
| 0.08 | 0.16 | 3.66 / 7.3 (−48 ±3) k8 | 4.30 / 8.6 (−39 ±3) k12 | 4.04 / 8.1 (−43 ±3) k12 | 0.68 / 1.4 (−90 ±4) k12 | 4.65 / 9.3 (−34 ±2) k12 | 4.22 / 8.4 (−40 ±3) k12 |
| 0.08 | 0.25 | 3.22 / 6.4 (−54 ±3) k7 | 3.62 / 7.2 (−49 ±3) k10 | 3.38 / 6.7 (−52 ±3) k12 | 0.45 / 0.9 (−94 ±4) k12 | 4.07 / 8.1 (−42 ±3) k12 | 3.51 / 7.0 (−50 ±3) k12 |
| 0.14 | 0.05 | 3.90 / 7.8 (−31 ±3) k11 | 4.23 / 8.3 (−25 ±3) k12 | 4.13 / 8.2 (−26 ±2) k12 | 1.01 / 2.0 (−82 ±5) k12 | 4.57 / 9.0 (−19 ±2) k12 | 4.18 / 8.3 (−26 ±3) k12 |
| 0.14 | 0.10 | 3.00 / 6.0 (−47 ±3) k8 | 3.43 / 6.8 (−39 ±3) k12 | 3.26 / 6.5 (−42 ±3) k12 | 0.55 / 1.1 (−90 ±4) k12 | 3.78 / 7.5 (−33 ±3) k12 | 3.25 / 6.5 (−42 ±3) k12 |
| 0.14 | 0.16 | 2.59 / 5.2 (−54 ±3) k7 | 2.87 / 5.7 (−49 ±3) k10 | 2.68 / 5.3 (−52 ±3) k12 | 0.37 / 0.7 (−93 ±4) k12 | 3.16 / 6.3 (−44 ±3) k12 | 2.66 / 5.3 (−53 ±3) k12 |
| 0.14 | 0.25 | 2.25 / 4.5 (−60 ±3) k6 | 2.46 / 4.9 (−56 ±3) k8 | 2.22 / 4.4 (−60 ±3) k8 | 0.25 / 0.5 (−96 ±3) k12 | 2.61 / 5.2 (−54 ±3) k12 | 2.29 / 4.6 (−59 ±3) k12 |
| 0.20 | 0.05 | 2.98 / 5.9 (−40 ±2) k10 | 3.33 / 6.6 (−33 ±3) k12 | 3.26 / 6.5 (−34 ±2) k12 | 0.70 / 1.4 (−86 ±4) k12 | 3.57 / 7.1 (−28 ±2) k12 | 3.16 / 6.3 (−36 ±2) k12 |
| 0.20 | 0.10 | 2.31 / 4.6 (−53 ±2) k7 | 2.55 / 5.1 (−49 ±3) k12 | 2.49 / 4.9 (−50 ±3) k12 | 0.37 / 0.7 (−93 ±3) k12 | 2.83 / 5.6 (−43 ±2) k12 | 2.35 / 4.7 (−53 ±3) k12 |
| 0.20 | 0.16 | 1.97 / 3.9 (−60 ±2) k6 | 2.11 / 4.2 (−58 ±2) k10 | 2.00 / 4.0 (−60 ±2) k8 | 0.27 / 0.5 (−95 ±3) k12 | 2.26 / 4.5 (−54 ±2) k12 | 1.95 / 3.9 (−61 ±2) k12 |
| 0.20 | 0.25 | 1.69 / 3.4 (−66 ±2) k5 | 1.82 / 3.6 (−63 ±2) k7 | 1.65 / 3.3 (−67 ±2) k8 | 0.19 / 0.4 (−96 ±3) k12 | 1.85 / 3.7 (−63 ±2) k12 | 1.68 / 3.3 (−66 ±2) k12 |

**Coin (one draw per red batch), W1**

| q | m | F1 alone | F1u | F2 halve | F3 as is | F3b | F4 bisect |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.05 | 7.04 / 14.0 (−16 ±2) k12 | 7.53 / 15.0 (−10 ±2) k12 | 8.09 / 16.1 (−3 ±2) k12 | 8.20 / 16.3 (−2 ±1) k12 | 7.94 / 15.8 (−5 ±2) k12 | 8.06 / 16.0 (−3 ±1) k12 |
| 0.05 | 0.10 | 6.13 / 12.2 (−27 ±3) k12 | 6.67 / 13.3 (−20 ±2) k12 | 7.74 / 15.4 (−7 ±2) k12 | 7.89 / 15.7 (−6 ±1) k12 | 7.45 / 14.9 (−11 ±2) k12 | 7.71 / 15.4 (−8 ±2) k12 |
| 0.05 | 0.16 | 5.24 / 10.4 (−37 ±3) k12 | 5.94 / 11.8 (−29 ±3) k12 | 7.29 / 14.5 (−13 ±2) k12 | 7.70 / 15.3 (−8 ±2) k12 | 6.95 / 13.8 (−17 ±2) k12 | 7.25 / 14.4 (−13 ±2) k12 |
| 0.05 | 0.25 | 4.35 / 8.7 (−48 ±3) k10 | 5.12 / 10.2 (−39 ±3) k12 | 6.73 / 13.4 (−19 ±2) k12 | 7.19 / 14.3 (−14 ±2) k12 | 6.23 / 12.4 (−25 ±2) k12 | 6.73 / 13.4 (−19 ±2) k12 |
| 0.08 | 0.05 | 5.83 / 11.6 (−17 ±3) k12 | 6.17 / 12.2 (−12 ±2) k12 | 6.74 / 13.4 (−4 ±2) k12 | 6.82 / 13.5 (−3 ±2) k12 | 6.56 / 13.0 (−7 ±2) k12 | 6.69 / 13.3 (−5 ±2) k12 |
| 0.08 | 0.10 | 4.74 / 9.4 (−33 ±3) k12 | 5.48 / 10.9 (−22 ±3) k12 | 6.31 / 12.5 (−10 ±2) k12 | 6.63 / 13.1 (−6 ±2) k12 | 6.14 / 12.2 (−13 ±2) k12 | 6.34 / 12.6 (−10 ±2) k12 |
| 0.08 | 0.16 | 4.13 / 8.2 (−41 ±3) k11 | 4.78 / 9.5 (−32 ±3) k12 | 6.05 / 12.0 (−14 ±2) k12 | 6.41 / 12.7 (−9 ±2) k12 | 5.66 / 11.3 (−20 ±2) k12 | 6.01 / 11.9 (−15 ±2) k12 |
| 0.08 | 0.25 | 3.37 / 6.7 (−52 ±3) k9 | 4.07 / 8.1 (−42 ±3) k12 | 5.42 / 10.8 (−23 ±2) k12 | 5.93 / 11.8 (−16 ±2) k12 | 5.01 / 10.0 (−29 ±2) k12 | 5.52 / 11.0 (−22 ±2) k12 |
| 0.14 | 0.05 | 4.44 / 8.7 (−21 ±3) k12 | 4.94 / 9.7 (−12 ±2) k12 | 5.19 / 10.3 (−8 ±2) k12 | 5.43 / 10.6 (−3 ±2) k12 | 5.22 / 10.3 (−7 ±2) k12 | 5.36 / 10.6 (−5 ±2) k12 |
| 0.14 | 0.10 | 3.71 / 7.4 (−34 ±3) k12 | 4.33 / 8.5 (−23 ±2) k12 | 4.90 / 9.8 (−13 ±2) k12 | 5.24 / 10.3 (−7 ±2) k12 | 4.82 / 9.5 (−14 ±2) k12 | 4.90 / 9.8 (−13 ±2) k12 |
| 0.14 | 0.16 | 3.08 / 6.1 (−45 ±3) k12 | 3.78 / 7.5 (−33 ±3) k12 | 4.53 / 9.0 (−19 ±2) k12 | 5.03 / 9.9 (−10 ±2) k12 | 4.45 / 8.8 (−21 ±2) k12 | 4.55 / 9.1 (−19 ±2) k12 |
| 0.14 | 0.25 | 2.49 / 5.0 (−56 ±3) k9 | 3.15 / 6.3 (−44 ±3) k12 | 3.93 / 7.8 (−30 ±2) k12 | 4.64 / 9.1 (−17 ±2) k12 | 3.86 / 7.7 (−31 ±2) k12 | 4.11 / 8.2 (−27 ±2) k12 |
| 0.20 | 0.05 | 3.84 / 7.6 (−23 ±2) k12 | 4.29 / 8.5 (−14 ±2) k12 | 4.56 / 9.1 (−8 ±1) k12 | 4.77 / 9.5 (−4 ±1) k12 | 4.56 / 9.1 (−8 ±1) k12 | 4.66 / 9.3 (−6 ±1) k12 |
| 0.20 | 0.10 | 3.17 / 6.3 (−36 ±3) k12 | 3.76 / 7.5 (−24 ±2) k12 | 4.22 / 8.4 (−15 ±1) k12 | 4.63 / 9.2 (−7 ±1) k12 | 4.19 / 8.3 (−16 ±1) k12 | 4.31 / 8.6 (−13 ±1) k12 |
| 0.20 | 0.16 | 2.58 / 5.1 (−48 ±3) k12 | 3.22 / 6.4 (−35 ±2) k12 | 3.80 / 7.6 (−24 ±2) k12 | 4.37 / 8.7 (−12 ±1) k12 | 3.79 / 7.5 (−24 ±2) k12 | 4.00 / 8.0 (−20 ±2) k12 |
| 0.20 | 0.25 | 2.10 / 4.2 (−58 ±3) k12 | 2.72 / 5.4 (−45 ±3) k12 | 3.21 / 6.4 (−35 ±2) k12 | 4.04 / 8.0 (−19 ±1) k12 | 3.26 / 6.5 (−34 ±2) k12 | 3.37 / 6.7 (−32 ±2) k12 |

The coin model flatters F2, F3 and F4 because it assumes a retry of the same culprit is scannable 84% of the time.

**For scale.** Each cell is lvh / landings per hour / verifies per landing / good wait p50/p95 h / defective arrival-to-block p50/p95 h / wrongly blocked per 100 landings.

| cell | TODAY | U (draft 3) | U+S default | U+S corrected, m=0 |
|---|---|---|---|---|
| W1 q=0.05 | 2.06 / 4.1 / 0.83 / 15/34 / 19/51 / 0.19 | 2.85 / 5.7 / 0.69 / 11/27 / 17/34 / 0.08 | 6.59 / 13.2 / 0.21 / 5/14 / 4/13 / 0 | 8.35 / 16.6 / 0.21 / 3/13 / 3/11 / 0 |
| W1 q=0.08 | 1.57 / 3.1 / 1.12 / 20/42 / 25/57 / 0.22 | 2.12 / 4.2 / 0.88 / 15/30 / 19/34 / 0.24 | 5.24 / 10.5 / 0.26 / 6/16 / 6/15 / 0 | 7.04 / 13.9 / 0.26 / 4/15 / 3/12 / 0 |
| W1 q=0.14 | 1.11 / 2.2 / 1.61 / 29/48 / 32/54 / 0.42 | 1.40 / 2.8 / 1.32 / 23/36 / 24/39 / 0.40 | 4.24 / 8.5 / 0.34 / 7/17 / 6/16 / 0 | 5.61 / 10.9 / 0.34 / 5/16 / 3/12 / 0 |
| W1 q=0.20 | 0.87 / 1.7 / 2.04 / 34/50 / 36/55 / 0.68 | 1.06 / 2.1 / 1.80 / 28/41 / 30/43 / 0.38 | 3.80 / 7.6 / 0.39 / 7/18 / 6/17 / 0 | 4.97 / 9.9 / 0.38 / 5/16 / 4/13 / 0 |
| W2 q=0.05 | 1.13 / 2.2 / 0.90 / 31/52 / 37/73 / 0.18 | 1.65 / 3.3 / 0.72 / 19/44 / 32/60 / 0.08 | 3.72 / 7.4 / 0.21 / 9/20 / 8/18 / 0 | 4.75 / 9.5 / 0.22 / 6/20 / 5/16 / 0 |
| W2 q=0.08 | 0.86 / 1.7 / 1.20 / 40/65 / 47/>60 / 0.21 | 1.22 / 2.4 / 0.92 / 28/51 / 36/58 / 0.27 | 2.86 / 5.7 / 0.28 / 11/24 / 10/22 / 0 | 4.03 / 7.9 / 0.27 / 8/23 / 6/18 / 0 |
| W2 q=0.14 | 0.60 / 1.2 / 1.72 / 58/81 / 61/>60 / 0.43 | 0.79 / 1.6 / 1.37 / 42/58 / 45/60 / 0.44 | 2.37 / 4.7 / 0.35 / 13/25 / 12/23 / 0 | 3.23 / 6.3 / 0.34 / 9/24 / 6/18 / 0 |
| W2 q=0.20 | 0.49 / 1.0 / 2.13 / 65/>60 / 66/>60 / 0.46 | 0.60 / 1.2 / 1.84 / 52/68 / 55/69 / 0.50 | 2.13 / 4.3 / 0.40 / 13/25 / 12/23 / 0 | 2.83 / 5.6 / 0.39 / 10/24 / 7/20 / 0 |

**Headline rows, m=0.16, sticky, W1.** "k, batch" is the k the rule picks and the mean number of un-cleared members per rule-built batch.

| q | fallback | lvh | land/h | verifies/landing | k, batch | good wait p50/p95 | time to block p50/p95 | wrong/100 | vs m=0 | vs F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | F1 | 4.79 ±0.16 | 9.5 | 0.36 | 10, 9.0 | 7/19 | 7/20 | 0 | −43 ±3 | — |
| 0.05 | F1u | 5.73 ±0.20 | 11.4 | 0.31 | 12, 10.4 | 5/18 | 5/19 | 0 | −31 ±3 | +20 ±3 |
| 0.05 | F2 | 5.46 ±0.20 | 10.9 | 0.34 | 12, 10.5 | 5/18 | 6/20 | 0 | −35 ±3 | +14 ±2 |
| 0.05 | F3 | 1.01 ±0.19 | 2.0 | 20.6 | 12, 11.8 | 14/>60 | 16/>60 | 0 | −88 ±4 | −79 ±5 |
| 0.05 | F3b | 6.20 ±0.20 | 12.3 | 0.29 | 12, 8.8 | 5/16 | 5/16 | 0.002 | −26 ±2 | +29 ±3 |
| 0.05 | F4 | 5.82 ±0.19 | 11.6 | 0.32 | 12, 10.8 | 5/17 | 5/19 | 0 | −30 ±3 | +21 ±3 |
| 0.08 | F1 | 3.66 ±0.11 | 7.3 | 0.49 | 8, 7.0 | 9/22 | 8/22 | 0 | −48 ±3 | — |
| 0.08 | F1u | 4.30 ±0.15 | 8.6 | 0.43 | 12, 9.7 | 7/21 | 7/21 | 0 | −39 ±3 | +18 ±3 |
| 0.08 | F2 | 4.04 ±0.14 | 8.1 | 0.48 | 12, 9.4 | 8/22 | 8/22 | 0 | −43 ±3 | +11 ±2 |
| 0.08 | F3 | 0.68 ±0.14 | 1.4 | 31.9 | 12, 11.9 | 22/>60 | 28/>60 | 0.010 | −90 ±4 | −81 ±5 |
| 0.08 | F3b | 4.65 ±0.15 | 9.3 | 0.40 | 12, 7.7 | 6/19 | 6/19 | 0.003 | −34 ±2 | +27 ±3 |
| 0.08 | F4 | 4.22 ±0.16 | 8.4 | 0.46 | 12, 9.9 | 7/22 | 7/23 | 0 | −40 ±3 | +15 ±3 |
| 0.14 | F1 | 2.59 ±0.09 | 5.2 | 0.72 | 7, 6.1 | 12/26 | 11/25 | 0 | −54 ±3 | — |
| 0.14 | F1u | 2.87 ±0.11 | 5.7 | 0.66 | 10, 8.0 | 10/25 | 10/25 | 0 | −49 ±3 | +11 ±2 |
| 0.14 | F2 | 2.68 ±0.12 | 5.3 | 0.74 | 12, 8.4 | 11/26 | 11/27 | 0 | −52 ±3 | +4 ±2 |
| 0.14 | F3 | 0.37 ±0.08 | 0.7 | 55.0 | 12, 11.9 | 38/>60 | 53/>60 | 0 | −93 ±4 | −86 ±5 |
| 0.14 | F3b | 3.16 ±0.13 | 6.3 | 0.62 | 12, 6.8 | 9/24 | 8/24 | 0.002 | −44 ±3 | +22 ±3 |
| 0.14 | F4 | 2.66 ±0.12 | 5.3 | 0.76 | 12, 8.9 | 11/27 | 11/28 | 0 | −53 ±3 | +3 ±3 |
| 0.20 | F1 | 1.97 ±0.06 | 3.9 | 0.96 | 6, 5.2 | 15/29 | 14/28 | 0 | −60 ±2 | — |
| 0.20 | F1u | 2.11 ±0.09 | 4.2 | 0.92 | 10, 7.9 | 14/30 | 13/30 | 0 | −58 ±2 | +7 ±3 |
| 0.20 | F2 | 2.00 ±0.07 | 4.0 | 1.00 | 8, 6.2 | 14/31 | 14/31 | 0 | −60 ±2 | +2 ±1 |
| 0.20 | F3 | 0.27 ±0.04 | 0.5 | 20.8 | 12, 11.2 | >60/>60 | >60/>60 | 0 | −95 ±3 | −86 ±4 |
| 0.20 | F3b | 2.26 ±0.09 | 4.5 | 0.88 | 12, 6.1 | 12/28 | 12/28 | 0.006 | −54 ±2 | +15 ±3 |
| 0.20 | F4 | 1.95 ±0.08 | 3.9 | 1.04 | 12, 6.2 | 14/31 | 14/32 | 0 | −61 ±2 | −1 ±2 |

**Headline rows, m=0.16, coin, W1**

| q | fallback | lvh | land/h | verifies/landing | k, batch | good wait p50/p95 | time to block p50/p95 | wrong/100 | vs m=0 | vs F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | F1 | 5.24 ±0.17 | 10.4 | 0.33 | 12, 10.2 | 6/18 | 6/18 | 0 | −37 ±3 | — |
| 0.05 | F2 | 7.29 ±0.19 | 14.5 | 0.25 | 12, 10.3 | 4/14 | 4/13 | 0 | −13 ±2 | +39 ±3 |
| 0.05 | F3 | 7.70 ±0.21 | 15.3 | 0.24 | 12, 10.3 | 4/14 | 3/12 | 0 | −8 ±2 | +47 ±4 |
| 0.05 | F4 | 7.25 ±0.20 | 14.4 | 0.25 | 12, 10.3 | 4/14 | 4/13 | 0 | −13 ±2 | +39 ±4 |
| 0.08 | F1 | 4.13 ±0.13 | 8.2 | 0.43 | 11, 8.9 | 7/22 | 7/21 | 0 | −41 ±3 | — |
| 0.08 | F2 | 6.05 ±0.18 | 12.0 | 0.31 | 12, 9.5 | 5/16 | 4/14 | 0 | −14 ±2 | +46 ±5 |
| 0.08 | F3 | 6.41 ±0.18 | 12.7 | 0.29 | 12, 9.5 | 5/16 | 4/13 | 0.001 | −9 ±2 | +55 ±5 |
| 0.08 | F4 | 6.01 ±0.17 | 11.9 | 0.31 | 12, 9.5 | 5/16 | 4/14 | 0 | −15 ±2 | +46 ±5 |
| 0.14 | F1 | 3.08 ±0.11 | 6.1 | 0.59 | 12, 8.5 | 9/24 | 8/22 | 0 | −45 ±3 | — |
| 0.14 | F2 | 4.53 ±0.09 | 9.0 | 0.42 | 12, 7.6 | 6/18 | 5/16 | 0 | −19 ±2 | +47 ±3 |
| 0.14 | F3 | 5.03 ±0.14 | 9.9 | 0.38 | 12, 8.6 | 6/18 | 4/15 | 0 | −10 ±2 | +63 ±5 |
| 0.14 | F4 | 4.55 ±0.11 | 9.1 | 0.42 | 12, 7.8 | 6/18 | 5/17 | 0 | −19 ±2 | +47 ±4 |
| 0.20 | F1 | 2.58 ±0.09 | 5.1 | 0.72 | 12, 8.2 | 11/26 | 9/24 | 0 | −48 ±3 | — |
| 0.20 | F2 | 3.80 ±0.09 | 7.6 | 0.51 | 12, 6.7 | 7/19 | 6/18 | 0 | −24 ±2 | +47 ±4 |
| 0.20 | F3 | 4.37 ±0.10 | 8.7 | 0.44 | 12, 7.7 | 6/18 | 5/16 | 0 | −12 ±1 | +70 ±5 |
| 0.20 | F4 | 4.00 ±0.09 | 8.0 | 0.49 | 12, 7.2 | 7/19 | 5/17 | 0 | −20 ±2 | +55 ±4 |

**W2 (median 75 min), m=0.16.** Each cell is lvh / landings per hour (Δ% vs corrected m=0; Δ% vs F1).

| model | q | F1 | F1u | F2 | F3 | F3b | F4 |
|---|---|---|---|---|---|---|---|
| sticky | 0.05 | 2.70 / 5.4 (−43) | 3.20 / 6.4 (−33; +19 ±4) | 3.08 / 6.1 (−35; +14 ±4) | 0.75 / 1.5 (−84) | 3.44 / 6.9 (−28; +28 ±4) | 3.31 / 6.6 (−30; +23 ±4) |
| sticky | 0.08 | 2.05 / 4.1 (−49) | 2.42 / 4.8 (−40; +19 ±4) | 2.29 / 4.6 (−43; +12 ±3) | 0.43 / 0.9 (−89) | 2.71 / 5.4 (−33; +32 ±4) | 2.42 / 4.8 (−40; +18 ±3) |
| sticky | 0.14 | 1.48 / 2.9 (−54) | 1.60 / 3.2 (−51; +8 ±3) | 1.53 / 3.0 (−53; +4 ±3) | 0.26 / 0.5 (−92) | 1.82 / 3.6 (−44; +23 ±4) | 1.54 / 3.1 (−52; +4 ±3) |
| sticky | 0.20 | 1.12 / 2.2 (−60) | 1.19 / 2.4 (−58; +6 ±3) | 1.14 / 2.3 (−60; +1 ±3) | 0.16 / 0.3 (−94) | 1.31 / 2.6 (−54; +17 ±4) | 1.14 / 2.3 (−60; +2 ±3) |
| coin | 0.05 | 3.03 / 6.0 (−36) | 3.44 / 6.8 (−28; +13 ±4) | 4.15 / 8.3 (−13; +37 ±4) | 4.38 / 8.7 (−8; +44 ±5) | 3.98 / 7.9 (−16; +31 ±4) | 4.16 / 8.3 (−12; +37 ±4) |
| coin | 0.08 | 2.35 / 4.7 (−42) | 2.86 / 5.7 (−29; +22 ±4) | 3.47 / 6.9 (−14; +48 ±4) | 3.62 / 7.2 (−10; +54 ±5) | 3.24 / 6.4 (−20; +38 ±4) | 3.48 / 6.9 (−14; +48 ±4) |
| coin | 0.14 | 1.85 / 3.7 (−43) | 2.26 / 4.5 (−30; +22 ±5) | 2.57 / 5.1 (−21; +39 ±4) | 2.86 / 5.6 (−11; +55 ±5) | 2.54 / 5.0 (−21; +38 ±5) | 2.64 / 5.2 (−18; +43 ±4) |
| coin | 0.20 | 1.50 / 3.0 (−47) | 1.88 / 3.7 (−34; +25 ±5) | 2.18 / 4.3 (−23; +45 ±5) | 2.54 / 5.0 (−10; +69 ±5) | 2.15 / 4.3 (−24; +43 ±4) | 2.25 / 4.5 (−21; +50 ±5) |

- **Against 2.4 arrivals/h.** At W2, q=0.20, sticky, F1, F2 and F4 land 2.2–2.3/h and F1u 2.4/h; only F3b clears it (2.6/h). F3 as-is is below 2.4/h in every sticky cell except q=0.05 at m ≤ 0.10. Every other fallback row is above it.
- **Wrongly blocked.** F1, F1u, F2 and F4 block no sound request in any scan-on row. F3 and F3b block at most 0.010 and 0.006 per 100 landings, all by elimination. I did not trace individual cases; they vanish at e=0 with no main-red. No defective request landed in any of the 620 rows.
- **Why halving does not pay under sticky.** At q=0.14, m=0.16, F2 and F4 produce 101 and 116 unscannable reds per run against F1's 45, and each tears down what is stacked behind it (torn 24% vs 20%). Smaller caps do not rescue them: at caps 6/24 and 8/24 F2 and F4 are no better than at 12/24 (a 13–14% loss at 6/24 at q=0.08, within ±3 at 8/24 at q=0.14).
- **The brief's envelope.** My cost code reproduces 2.24 → 3.65 for F1 at q=0.14, m=0.16. The envelope predicted −39%; the simulation gives −45 ±3 (coin) and −54 ±3 (sticky).
- **Other cells.** All measures for every row are in `results2_F_compact.txt`.

## (b) Corrections, each alone on the default U+S

Δ% in lvh vs default U+S (prefix scan, k+1 trees, on the host, m=0).

| change | q=0.08 | q=0.14 |
|---|---|---|
| member-alone scan (n+2 trees) | −4 ±1 | −3 ±1 |
| T1 confirm tree per culprit | −4 ±1 | −6 ±1 |
| T2 only the first culprit blocked (moot) | −19 ±1 | −36 ±2 |
| T3 scan off the hosts, one resource | +59 ±2 | +61 ±2 |
| T3, rule still charges c_t | +53 ±2 | +51 ±1 |
| T3, two resources | +62 ±2 | +65 ±2 |
| T4 no evidence 1% | −11 ±2 | −8 ±1 |
| n+2, T1, T4, scan on the host | −16 ±2 | −15 ±2 |
| corrected (n+2, T1, T3×1, T4) | +34 ±4 | +32 ±4 |
| corrected with two scan resources | +37 ±4 | +40 ±4 |
| corrected with T2 also on | +7 ±3 | −16 ±2 |

**Does one scan resource keep up?** Yes at every q: utilisation stays below 1 and the queue stays bounded (at most 6 jobs). Members are held a long time, though.

| q | utilisation | queue wait p50/p95 h | members held p50/p95 h | with two resources: wait p95, lvh |
|---|---|---|---|---|
| 0.05 | 0.51 | 0.16 / 1.14 | 0.87 / 1.79 | 0.06 h, +1% |
| 0.08 | 0.64 | 0.41 / 1.41 | 1.03 / 2.06 | 0.15 h, +2% |
| 0.14 | 0.74 | 0.80 / 1.73 | 1.43 / 2.38 | 0.18 h, +6% |
| 0.20 | 0.69 | 0.39 / 1.60 | 0.99 / 2.17 | 0.04 h, +2% |

With m=0.16 sticky and F1 the scan resource is lighter (utilisation 0.23–0.29, wait p95 0.2–0.5 h), and a second resource changes the fallback comparison by ±2%.

## (c) What I would write into a contract

1. Write F1u: an unscannable red batch sends its un-cleared members to verify alone, each tagged with the batch; when one of them blocks, the rest return as fresh. If the red-observation ledger of draft 3 is being built anyway, F3b is 2–12% better still (sticky).
2. Do not write F3 as it stands. Its cost rule assumes the next scan will resolve the set, so with a sticky culprit it re-dispatches the same members (55 verifies per landing at q=0.14).
3. At the measured m=0.16 (sticky), F2's extra state buys +14/+11/+4/+2% over F1 and is 5–6% behind F1u at every q. It is not worth it. F4 is no better at q ≥ 0.14.
4. F1 alone is within 10% of F2 at every q for m ≤ 0.10, and at q ≥ 0.14 for every m (sticky). Against F1u it is 8–12% behind even at m=0.05. Under the coin model F2 leads F1 by 15–19% already at m=0.05.
5. The cost is in m, not the fallback: under sticky the share of red 12-batches that cannot be scanned is 20/23/28/35% at q = 0.05/0.08/0.14/0.20 for m=0.16, and each 0.05 of m costs about 8–13 points.

## (d) Closed forms used by the cost rule

Notation: p is a fresh member's prior; r_j = 1−(1−p)^j; v_j = 1−(1−p·m)^j; a = ⌈j/2⌉, b = j−a; a0 = 1/12 + ν is the rider's going cost; ν = 0.01 (T4); c is the rule's tree cost (0 when the scan is off the hosts). n is the number of un-cleared members, k all members, a_f the going cost of a fresh member.

- **Batch cost.** C(S) = 1 + P_red·[(1−u)·SR + u·(2c + FB(n) + (k−n)·a0)] + ν·k, with P_red = 1 − g(S).
- **Scan resolves it.** SR = c·(k+2+nb) + Σ_i (1 − p_i/P_red)·a0, where nb = Σ p_i/P_red is the expected culprits (T1). With T2: c·(k+2+1) and an extra (nb−1) verifies.
- **Unscannable share u.** Coin: m. Sticky: (1 − Π(1 − p_i·m)) / P_red. F3 as-is: 0 (unchanged).
- **F1:** FB(n) = n; FB(1) = 0.
- **F1u:** FB(n) = B̄ + 1 + (n−1−B̄)·a_f, with B̄ = Σ_j (j−1)(1−p)^(j−1)·p / r_n the sound members ahead of the first culprit.
- **F2, coin:** X(1) = 0; X(j) = 2 + Σ_{h∈{a,b}, h>1} (r_h/r_j)·[m·(2c + X(h)) + (1−m)·SR(h)].
- **F2, sticky:** X(j) = 2 + Σ_{h>1} [v_h·(2c + X(h)) + (r_h − v_h)·v_{j−h}·SR(h)] / v_j.
- **F2 or F2′ with no scan (m=1):** X(j) = 2 + (r_a·X(a) + r_b·X(b))/r_j. K(j) = 1 + (r_a/r_j)·[K′(a) + Y(b)] + (1 − r_a/r_j)·K″(b), with K′(1)=0, K′(a)=K(a); Y(1)=1, Y(b)=1 + r_b·K(b); K″(1)=1, K″(b)=K(b).
- **F4, coin:** B(j) = 1 + (r_a/r_j)·[b·a_f + (a>1: m·(2c+B(a)) + (1−m)·SR(a))] + ((1−r_a)·r_b/r_j)·G(b), with G(1)=1, G(b)=B(b).
- **F4, sticky:** B(j) = 1 + (v_a/v_j)·[b·a_f + (a>1: 2c+B(a))] + ((r_a−v_a)·v_b/v_j)·[SR(a) + b·a_f] + ((1−r_a)·v_b/v_j)·G(b).
- **Going cost.** a(p) = min over k ≤ 12 of C(k fresh at p)/k, by fixed point where a_f appears.
- **Scan predicate (X2).** Scan iff (k+2)·c_t < U(n), with U = n (F1), X at m=1 (F2), n·a_U(p/r_n) (F3b, plain U's going cost). A batch that would not be scanned is priced 1 + P_red·U(n).

Closed-form follow-up of an unscannable red 12-batch at q=0.14, m=0.16: F1 12.0; F1u 6.4 (coin) / 6.8 (sticky); F2 3.1 / 8.1; F4 2.9 / 4.9.

## (e) Checks

- **m=0.** With the corrections off, all five fallbacks reproduce default U+S run for run (20 of 20 seeds at q=0.08 and at q=0.14). With the corrections on they are identical to each other (20 of 20).
- **e=0, no main-red.** 1,200 runs over 50 variants: no sound request blocked or convicted, no defective one landed. The variants cover every fallback, both models, m 0.16 and 0.25, T2, on-host scan, prefix-binary scan, the predicates, scan off and never-scan.
- **Scan off entirely.** F1 lands within 1–2% (±1) of TB at W1 and 2–3% at W2, and picks k = 5/4/3/3, the table depth. TB is T+ (fixes (i) and (ii), marks kept across restarts) with pool order and static table depth. It is not run for run: U also remembers a batch that passed and was torn down, and keeps "alone" through a main-red bounce. T+ is 3–4% below F1 and TODAY 10–13% below.
- **Process count.** Runs used 4 workers; for a few minutes at a time a fifth single process (smoke tests, tables) ran beside them.

## X1. No scan at all

Engine: every red takes the fallback, no flake re-run, T4 off. lvh, with the k the rule picks in brackets.

| cell | TODAY | TB | F1 | F1u | F2 | F2′ | F4 | F3 plain U |
|---|---|---|---|---|---|---|---|---|
| W1 q=0.05 | 2.06 | 2.34 | 2.36 [5] | 2.73 [7] | 2.32 [8] | 2.57 [8] | 2.75 [8] | 2.85 |
| W1 q=0.08 | 1.57 | 1.79 | 1.82 [4] | 2.02 [5] | 1.67 [8] | 1.85 [8] | 1.84 [8] | 2.12 |
| W1 q=0.14 | 1.11 | 1.24 | 1.26 [3] | 1.35 [4] | 1.14 [4] | 1.20 [4] | 1.22 [4] | 1.40 |
| W1 q=0.20 | 0.87 | 0.95 | 0.97 [3] | 0.97 [4] | 0.91 [2] | 0.88 [4] | 0.86 [4] | 1.06 |
| W2 q=0.08 | 0.86 | 1.01 | 1.03 | 1.15 | 0.95 | 1.05 | 1.06 | 1.22 |
| W2 q=0.14 | 0.60 | 0.70 | 0.72 | 0.77 | 0.65 | 0.68 | 0.69 | 0.79 |

Paired Δ% in lvh:

| cell | F3 over F1 | F3 over F2 | F3 over F2′ | F2′ over F2 | F2 over F1 | F1u over F1 | F1u vs F3 |
|---|---|---|---|---|---|---|---|
| W1 q=0.05 | +21 ±3 | +23 ±3 | +11 ±3 | +11 ±2 | −1 ±1 | +16 ±2 | −4 ±2 |
| W1 q=0.08 | +17 ±2 | +27 ±2 | +15 ±2 | +11 ±1 | −8 ±1 | +11 ±2 | −5 ±2 |
| W1 q=0.14 | +11 ±2 | +23 ±2 | +16 ±2 | +6 ±1 | −10 ±2 | +7 ±2 | −3 ±1 |
| W1 q=0.20 | +9 ±2 | +16 ±2 | +20 ±2 | −3 ±1 | −6 ±1 | 0 ±1 | −8 ±1 |
| W2 q=0.08 | +19 ±4 | +28 ±4 | +17 ±3 | +10 ±2 | −8 ±2 | +11 ±2 | −6 ±3 |
| W2 q=0.14 | +10 ±2 | +22 ±3 | +17 ±2 | +4 ±1 | −10 ±2 | +7 ±2 | −3 ±2 |

- Leaving the statistical design out costs 9–21% against each-alone and 16–28% against halving.
- Halving is worse than each-alone at q ≥ 0.08 when there is no scan; the green-half inference recovers 4–11 points at q ≤ 0.14 and loses 3 at q=0.20.
- F1u recovers most of the gap to F3 (3–8% behind).
- Wrongly blocked per 100 landings: F3 0.08–0.40, F1 0.14–0.34, halving 0.03–0.23. With no flake re-run a false failure on a single blocks.
- F2′ here holds the second half while the first is verified; dispatched side by side the inference cannot save a verify.

## X2. Break-even tree cost

Setup: m=0.16 sticky, corrected lane, one scan resource, trees one at a time. Figures are landings per hour. In lvh, always-scan beats never-scan at every c_t up to 0.80, because an off-host scan costs no verify time and idle hosts are not counted; at c_t ≥ 0.20 hosts idle 6–69%.

| q | fallback | never | always: 0.05 | 0.10 | 0.20 | 0.30 | 0.50 | 0.80 | break-even c_t |
|---|---|---|---|---|---|---|---|---|---|
| 0.08 | F1 | 3.61 | 7.29 | 7.21 | 6.62 | 5.51 | 3.87 | 2.69 | 0.57 |
| 0.08 | F2 | 3.26 | 8.06 | 7.68 | 6.10 | 4.80 | 3.37 | 2.40 | 0.53 |
| 0.08 | F3b | 4.14 | 9.25 | 8.85 | 6.87 | 5.22 | 3.57 | 2.50 | 0.43 |
| 0.14 | F1 | 2.50 | 5.15 | 5.08 | 4.47 | 3.63 | 2.48 | 1.73 | 0.50 |
| 0.14 | F2 | 2.23 | 5.34 | 5.09 | 4.09 | 3.28 | 2.25 | 1.54 | 0.51 |
| 0.14 | F3b | 2.75 | 6.28 | 5.85 | 4.58 | 3.58 | 2.50 | 1.68 | 0.45 |

F3 as-is loses to never-scan at both costs I ran (0.05 and 0.20): always-scan lands 0.7–1.8/h against plain U's 2.75–4.14/h.

**Predicate, with the cost rule knowing it** (landings per hour; in brackets the share of red batches scanned):

| q | fallback | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 | 0.80 |
|---|---|---|---|---|---|---|---|
| 0.08 | F1 | 7.30 (74%) | 7.21 | 6.75 | 5.77 | 4.27 (62%) | 3.93 (41%) |
| 0.08 | F2 | 8.04 (85%) | 7.59 | 6.40 | 5.16 | 4.10 (63%) | 3.26 (0%) |
| 0.08 | F3b | 9.22 (88%) | 8.83 | 7.04 | 5.64 | 4.32 (68%) | 4.11 (0%) |
| 0.14 | F1 | 5.20 (70%) | 5.13 | 4.70 | 3.99 | 3.25 (53%) | 2.97 (30%) |
| 0.14 | F2 | 5.35 (82%) | 5.17 | 4.28 | 3.56 | 2.93 (56%) | 2.23 (0%) |
| 0.14 | F3b | 6.20 (86%) | 5.83 | 4.82 | 3.92 | 3.20 (59%) | 2.60 (0%) |

- **Never worse?** In landings per hour, yes within the CI in 35 of 36 cells. The exception is F3b at q=0.14, c_t=0.80: −5 ±2 against never-scan. It beats always-scan by +5 to +14% at c_t 0.20–0.30 and by +10 to +72% at c_t ≥ 0.50.
- **In lvh** it is 7–22% below always-scan at c_t=0.50 and 34–58% below at 0.80, because always-scan idles the hosts.
- **The rule must know the predicate.** With the predicate at the red only, c_t=0.80 lands 6–14% below never-scan: the rule still sizes batches for a scan that never runs.
- **Queue-aware predicate** (count the backlog as trees to run): worse than the plain one for F1 at every c_t ≥ 0.20; better for F3b at c_t ≥ 0.30 (6.05 vs 5.64 at q=0.08, c_t=0.30). Waiting is cheap while the hosts have other work.
- **Prefix binary search** overtakes the member-alone scan in landings per hour at c_t ≈ 0.20 with F1 and ≈ 0.15 with F2. At c_t=0.30, F1 lands 6.43 vs 5.51 (q=0.08) and 4.15 vs 3.63 (q=0.14). In lvh it never overtakes (3.31 vs 3.43 at q=0.08, c_t=0.30). At c_t=0.05 it is 9–16% behind.

## X3. Concurrency inside a scan (q=0.14, corrected lane, m=0)

| c_t | scans at a time × trees side by side | lvh | land/h | hosts idle | queue wait p50/p95 h | members held p50/p95 h | scan utilisation |
|---|---|---|---|---|---|---|---|
| 0.05 | 1 × 1 | 5.61 | 10.9 | 3% | 0.80 / 1.73 | 1.43 / 2.38 | 0.74 |
| 0.05 | 1 × 2 | 5.89 | 11.7 | 0% | 0.00 / 0.39 | 0.44 / 0.83 | 0.49 |
| 0.05 | 1 × 4 | 5.94 | 11.8 | 0% | 0.00 / 0.13 | 0.25 / 0.45 | 0.33 |
| 0.05 | 1 × 8 | 5.88 | 11.7 | 0% | 0.00 / 0.08 | 0.22 / 0.33 | 0.26 |
| 0.05 | 2 × 1 | 5.95 | 11.8 | 0% | 0.00 / 0.18 | 0.57 / 1.00 | 0.40 |
| 0.20 | 1 × 1 | 4.69 | 4.7 | 49% | 10.7 / 12.3 | 12.8 / 14.1 | 0.99 |
| 0.20 | 1 × 2 | 4.76 | 6.6 | 31% | 6.0 / 7.4 | 7.3 / 8.7 | 0.94 |
| 0.20 | 1 × 4 | 4.95 | 8.2 | 17% | 3.4 / 4.7 | 4.3 / 5.6 | 0.88 |
| 0.20 | 1 × 8 | 5.17 | 9.4 | 9% | 2.1 / 3.4 | 2.9 / 4.2 | 0.83 |
| 0.20 | 2 × 1 | 4.94 | 7.6 | 23% | 3.1 / 4.4 | 5.1 / 6.8 | 0.90 |
| 0.20 | 2 × 2 | 5.26 | 9.7 | 8% | 0.9 / 2.0 | 2.3 / 3.6 | 0.78 |
| 0.20 | 2 × 4 | 5.66 | 11.1 | 2% | 0.03 / 0.78 | 1.15 / 1.98 | 0.61 |

- **At c_t=0.05** one scan at a time is still the right bound once trees run side by side. Two trees give +5 ±2% lvh and cut the hold from 1.43/2.38 h to 0.44/0.83 h, which is better than two scans of one tree each. Good-request wait stays 5/15–16 h.
- **At c_t=0.20** it is not. One scan saturates at every concurrency tested (utilisation 0.83–0.99), and for the same number of slots two scans beat one: 2×4 lands 11.1/h against 1×8 at 9.4/h. The base, tip and confirm trees of each scan are serial.
- **With m=0.16 sticky and F1** concurrency moves landings per hour by +1–2% at c_t=0.05 and +12–16% at c_t=0.20; the hold falls from 0.32/0.72 h to 0.14/0.22 h at c_t=0.05.

## Simplifications that could change a ranking

- **[RANK] Sticky is all-or-nothing.** A tree holding any unscannable culprit, among its own members or the entries it is stacked on, learns nothing from the scan, even about a scannable culprit beside it.
- **[RANK] m is per culprit under sticky.** If the measured 16% was per red batch at today's 3–4 members, the per-culprit rate is about 0.14–0.15 and the m=0.16 rows are slightly pessimistic.
- **[RANK] The closed forms count verifies, not tear-down.** Fallback entries are dispatched stacked at the front of the pool under today's cascade. Under a narrowed cascade, or with fallback tests built on main, halving and bisecting would rank higher; I did not run that.
- **[RANK] The scan resource is separate from the verify hosts.** Its load on the workstation is not modelled, a scan survives a restart, and riders count as members in n+2.
- **[RANK] Under T3 the rule prices the scan at zero,** and lvh sees neither scan time nor idle hosts. At c_t ≥ 0.20 read landings per hour.
- **[RANK] F4 never blocks by elimination** (the last member verifies alone), and it releases the members behind a red half as fresh.
- **[RANK] A red on unresolved entries ahead is provisional** in F1, F1u, F2 and F4: its members are held until those entries resolve, and nobody is touched if one fails.
- **[RANK] An unscannable red single blocks at the head as today;** its two trees only load the scan resource.
- **[RANK] T4 sends riders alone too.** Under sticky a main-red episode is unscannable with probability m.
- **[RANK] The membership test stalls on a non-convex cost.** In X1, the never-scan rows and the predicate rows the rule may add 2–4 fresh members jointly (up to the table's k with the predicate). Without it F2′ picks 2 instead of 4 at q=0.14.
- **[RANK] Inherited from the earlier runs:** deterministic non-interacting defects, exact known-good, a pool that refills without limit.
- **[RANK] Coverage.** W2 was run at m=0.16 only; the cap, second-resource and rule-c_t sensitivities at q = 0.08 and 0.14 only.

## Files

Everything is under `/tmp/gate-10069/sim/`; the repo is unchanged.

- **Engine:** `lane.py` (`FBCost`, `UPolicy._red_no_scan`, `_fallback`, `want_scan`, `_cost`; `Engine._scan_apply`, `_scan_offhost`, `_no_evidence`, `_unscannable`); `lane_v5.py` is the frozen before.
- **Cells and runner:** `fbcfg.py`, `fbgrid.py` (plans `ref`, `tog`, `coin_lo/hi`, `sticky_lo/hi`, `w2`, `sens`, `f1u`, `x1`, `x2a`, `x2b`, `x2r`, `x3`).
- **Frozen copies the runs used:** `run6/`, `run6x/`, `run6y/`, `run6z/`. `consist6.py` shows the current engine matches them run for run on 11 rows × 10 seeds.
- **Tables:** `results2_F_compact.txt` (620 rows, every measure), built by `compactF.py`; `showMain.py`, `showX.py`, `showtog.py`, `showS.py`.
- **Raw aggregates:** `results/F_*.json`.
- **Checks:** `checks6.py`, `out_checks6.txt`, `repro_v5.py`.
