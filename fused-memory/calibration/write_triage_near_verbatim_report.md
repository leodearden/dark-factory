# Write-triage near-verbatim rule: what a `restated` write may differ by (task 7155, 2026-11-09)

This record is written by hand. Nothing regenerates it. It is the basis for
`src/fused_memory/server/near_verbatim.py` (`is_near_verbatim`), which decides
whether a write may be filed `restated` (a sighting child that adds nothing) rather than judged.
`tests/server/test_near_verbatim.py` re-runs the committed-fixture set (set 5 below) on every verify.

## Decision

A write is near-verbatim to a reference only when its lowercased token SEQUENCE is identical to the
reference's, where a token is a word or a single symbol: `re.findall(r'\w+|[^\w\s]', text.lower())`. Only
case and whitespace may differ. Symbols are tokens, so `>` vs `<` and `-5` vs `5` differ; order counts, so
two swapped numbers or actors differ.

**Revised 2026-11-18 on review (task 7155).** The rule as first shipped was two conjuncts: the same token
MULTISET, and a character ratio `difflib.SequenceMatcher(None, n(a), n(b), autojunk=False).ratio() >= 0.98`
over the texts lowercased with whitespace runs collapsed (`n()`, the measure soak seat A used). Review found
two defects in it. A long text that only reorders its tokens passed both conjuncts. In the 1,028-char
synthetic text of `tests/server/test_near_verbatim.py`, swapping "retries 3 times with a 5 second wait" to
"retries 5 times with a 3 second wait" scores 0.9981, and swapping the actors of "the merge worker runs before
the hook" scores 0.9883, so either would have been filed `restated` with no judge call. And such a text paid the quadratic character ratio on the
write path, synchronously: the review measured 2.2 s at 6.9k chars and 125 s for a word-shuffled 42k-char
text, all of it blocking the server's event loop. The ordered sequence check closes the reorder hole, has no free
parameter and runs in linear time.

## Measurements

The figures below were taken with the first-shipped two-conjunct rule. "Similarity" is its character ratio.
They carry over to the ordered rule, which passes a pair only if the pair has the same token multiset. Every
pair that passed the two-conjunct rule was an exact duplicate, and the ordered rule passes those too. A pair
could newly pass only if it had the same token sequence but a ratio below 0.98, i.e. it differed only in
whitespace beside a symbol. A full-text re-check of sets 1 and 2 on 2026-11-18 (1,822 pairs, from
`attaches_soak.json` and `parents.json`) found the same 2 pairs token-multiset-equal, token-sequence-equal
and equal after normalization. Set 5 is re-run under the ordered rule by the test on every verify.

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

- **Token-sequence equality has no free parameter.** It excludes the single-token-change class, a measurement
  update or a correction, which no ratio cut can exclude at every length: a k-char change in an n-char text
  scores about 1 - k/n. It also excludes every reordering, which a token multiset cannot see.
- **It gives up nothing observed.** In every measured set, a near-verbatim restatement was an exact re-write,
  and the ordered rule accepts all of them. The "small reorderings" the first-shipped rule tolerated were never
  observed among genuine restatements.
- **The cost is asymmetric, so the rule is strict.** A missed near-verbatim write costs one judge call and is
  filed `amended` (a redundant digest). A false near-verbatim hides a claim behind a sighting, whose text a
  reader of the parent never sees.

## Caveats

- **The first-shipped 0.98 cut was set after seeing sets 1-3.** Sets 4 and 5 were measured after it was
  chosen. The ordered rule has no cut to fit.
- **Spacing beside a symbol is ignored.** `x>5` and `x > 5` are the same token sequence and so near-verbatim
  (pinned in `tests/server/test_near_verbatim.py`, with the swapped-number and swapped-actor cases). No pair
  in sets 1, 2, 4 or 5 differs only that way; set 3 was not re-checked.
- **Set 3's text source may be truncated.** In `duplicates/journal_add_memory.jsonl`, 6,633 of the 7,673
  texts keyed by memory id are exactly 200 characters long (counted 2026-11-18), so a set-3 figure taken from
  that file may describe 200-character prefixes rather than whole texts.

## Evidence locations (main checkout, gitignored, read-only)

- `/home/leo/src/dark-factory/data/write-triage-soak-2026-10-19/correctness/attaches_soak.json` and
  `.../correctness/ratings/`: the soak attaches and their ratings (sets 1 and 2).
- `/home/leo/src/dark-factory/data/write-triage-soak-2026-10-19/team/A/report.md`: seat A's near-verbatim
  measure, finding (c).
- `/home/leo/src/dark-factory/data/write-triage-soak-2026-10-19/duplicates/journal_add_memory.jsonl`: the
  texts of the nearest-neighbour pairs (set 3).
- `/home/leo/src/dark-factory/data/reconciliation/write_journal.db`, opened `mode=ro`: the post-soak children
  and their parent texts (set 4).
