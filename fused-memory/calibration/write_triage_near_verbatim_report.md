# Write-triage near-verbatim rule: what a `restated` write may differ by (task 7155, 2026-11-09)

This record is written by hand. Nothing regenerates it. It is the basis for
`src/fused_memory/server/near_verbatim.py` (`is_near_verbatim`, `NEAR_VERBATIM_MIN_RATIO`), which decides
whether a write may be filed `restated` (a sighting child that adds nothing) rather than judged.
`tests/server/test_near_verbatim.py` re-runs the committed-fixture set (set 5 below) on every verify.

## Decision

A write is near-verbatim to a reference only when both hold:

1. **Same tokens.** The multiset of lowercased tokens is identical, where a token is a word or a single
   symbol: `re.findall(r'\w+|[^\w\s]', text.lower())`. Symbols are tokens, so `>` vs `<` and `-5` vs `5`
   differ.
2. **Near-identical order.** `difflib.SequenceMatcher(None, n(a), n(b), autojunk=False).ratio() >= 0.98`,
   where `n()` lowercases and collapses whitespace runs to one space. This is the measure soak seat A used.

The token check runs first and short-circuits, because the character ratio is slow on long texts.

## Measurements

All figures were taken with exactly this rule. "Similarity" is the character ratio of conjunct 2.

| set | pairs | pass the rule | highest non-identical similarity |
|---|---|---|---|
| 1. in-sample: rated `restated` soak attaches vs the filed parent | 40 | 0 (none token-equal) | 0.726 (seat A, max over parent and cosine winner: 0.755) |
| 2. the other soak attaches (85 unrated restated, 1,224 amended, 473 contested) | 1,782 | 2, both exact duplicates | 0.756, a measurement update with new timings |
| 3. nearest-neighbour pairs at cosine >= t_high (140 base window 2026-10-04..11, 122 soak) | 262 | 3, all exact duplicates | 0.9699, a correction (see below) |
| 4. post-soak triage children 2026-10-19 09:00 .. 2026-11-09 with parent text (882 sightings, 1,938 amendments, 716 contested) | 3,536 | 1 token-equal, an exact duplicate | — |
| 5. committed fixtures: non-canonical vs canonical across `tests/fixtures/write_triage_calibration*.jsonl` (91 labelled duplicate), plus rated SAME/SUBSUMED pairs in `tests/fixtures/write_triage_pair_verdicts_seed.jsonl` | 272 + 24 | 0 | 0.668 |

The 0.9699 pair in set 3 is a write that prepends "Correction to the earlier entries this replaces: ... is no
longer unrouted." to its neighbour's text. A similarity-only rule at any threshold <= 0.97 would have filed that
correction as `restated`.

In this corpus, a near-verbatim restatement is an exact re-write. Semantic duplicates are paraphrases at
similarity <= 0.76.

## Basis

- **The token conjunct has no free parameter.** It excludes the single-token-change class, a measurement update
  or a correction, which no ratio cut can exclude at every length: a k-char change in an n-char text scores
  about 1 - k/n.
- **0.98 is the smallest round value above the highest observed non-identical similarity (0.9699)**, so the
  ratio alone also excludes every observed non-identical pair. Each conjunct alone rejects them all (defence in
  depth).
- **The cost is asymmetric, so the rule is strict.** A missed near-verbatim write costs one judge call and is
  filed `amended` (a redundant digest). A false near-verbatim hides a claim behind a sighting, whose text a
  reader of the parent never sees.

## Caveats

- **0.98 was set after seeing sets 1-3.** Sets 4 and 5 were measured after it was chosen.
- **Residual risk: a long text that only reorders tokens.** It passes the token check, and a small reordering
  of a long text can stay above 0.98. A reordering that swaps the actors of a short sentence scores below it
  (pinned in `tests/server/test_near_verbatim.py`).

## Evidence locations (main checkout, gitignored, read-only)

- `/home/leo/src/dark-factory/data/write-triage-soak-2026-10-19/correctness/attaches_soak.json` and
  `.../correctness/ratings/`: the soak attaches and their ratings (sets 1 and 2).
- `/home/leo/src/dark-factory/data/write-triage-soak-2026-10-19/team/A/report.md`: seat A's near-verbatim
  measure, finding (c).
- `/home/leo/src/dark-factory/data/write-triage-soak-2026-10-19/duplicates/journal_add_memory.jsonl`: the
  texts of the nearest-neighbour pairs (set 3).
- `/home/leo/src/dark-factory/data/reconciliation/write_journal.db`, opened `mode=ro`: the post-soak children
  and their parent texts (set 4).
