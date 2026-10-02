# Rater brief: the subject-level attach scale

This is the scale used for every verdict in `verdicts/verdict_cache.jsonl`: 4 Opus raters (medium effort) plus a Sonnet (high effort) second rater. Inter-rater κ was 0.76–0.82, and blind re-rates of anchor items agreed 9/10. New ratings must use it **verbatim**, or they are not comparable with the cache.

Each item is a blinded (CHILD, PARENT) pair. CHILD is the incoming write (the fixture entry), and PARENT is the record it was attached to. Raters must not see which system produced the pair, or any key file.

## Question for each pair

Judge CHILD's central claim against PARENT's central claim or claims. Does CHILD belong filed under PARENT as a restatement, extension or correction of it?

- **SAME**: CHILD states the same claim as PARENT (a restatement or rediscovery).
- **EXTENDS**: CHILD is about the same claim and adds a detail, example, condition, cause or fix, without changing the claim.
- **SUBSUMED**: PARENT is a broader note whose claims include CHILD's claim.
- **CORRECTS**: CHILD addresses the same claim or subject as PARENT and says it is wrong, outdated or different, explicitly or implicitly. This covers a correction or contradiction of PARENT's own claim.
- **RELATED**: same topic or area, but CHILD's central claim is about a different subject than PARENT's. CHILD is not a restatement, extension or correction of it. Sharing only a secondary sub-point is RELATED.
- **UNRELATED**: different topic.
- **UNCLEAR**: the texts do not let you decide, for example when a truncation falls at the crucial point. Use it sparingly and say why.

Belongs = SAME, EXTENDS, SUBSUMED or CORRECTS. Wrong (a misfile) = RELATED or UNRELATED.

## Procedure

Read items in small batches. Read each text in full, up to its length; never judge from a short prefix. Judge each item independently. Write one JSON line per item: `{"id","verdict","reason"}`, where `reason` is one sentence naming the two central claims.

Texts were capped at 4,000 characters with a `…[truncated, N chars total]` marker. Keep the same cap.

## Design used so far

- Primary raters: Opus, medium effort, one shard of about 44–55 items each.
- Second rater: Sonnet, high effort, on a random overlap of 30–60 items, for κ.
- Blind anchors: about 10 already-rated pairs mixed into new shards, including some previously rated wrong, to detect drift between batches.
- Pairs already in `verdict_cache.jsonl` (same entry_id, target_id) are reused, not re-rated.
