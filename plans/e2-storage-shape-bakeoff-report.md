# E2 storage-shape bake-off

Arbitration experiment for `docs/prds/memory-metadata-vocabulary.md` D9/D10 (task 3199, PRD leaf ζ), implementing `plans/memory-subsystem-eval-design.md` §5 E2.  This artifact is the signal gate leaf **η** reads: the choice between δ-as-default and peers-as-default is made from the table below.

## How to read this table

Every metric here is **rank/set-based** — ranks and set membership (present-in-top-k), never absolute cosine — per eval-design §1: wording and embedding-config drift move the score scale wholesale, so thresholds on raw cosine do not survive a re-measurement while ranks do.

The ONE exception is the last column.  `guard matched (replay)` is a **threshold replay**: the production near-duplicate selector *is* an absolute-threshold selector, so replaying it is the only way to answer "would the guard have fired?".  Do not trend that column across an embedder or config change — trend `guard candidate present`, which is rank-based and asks the discriminating question (was a true cluster sibling in front of the guard at all?).

A `—` cell is **no measurement**, not a measured zero.

`median canonical rank` carries its denominator as `(n=found/candidates)`.  The median is over the queries where the canonical surfaced AT ALL, so without that suffix an arm that almost never finds the canonical prints the best rank in the table — scored on the handful of queries where it did.  Rank is measured over the full fetch depth, not the k=5 read window, so "outside top-5" and "absent entirely" stay distinguishable.

`pin changed window` is the diagnostic that makes the `+pin` rows readable.  Every variant is scored over a window of the SAME size (k), so an additive pin can only pay off where a read-side transform left headroom in that budget — under grouping, which collapses the window.  A `+pin` row identical to its twin at `0.00` means the pin never fired, which is a different finding from "the pin does not help".  `—` on a pin-off row means the question was never asked.

## Decision table

| arm | claim recall@5 | claim recall@10 | canonical in top-5 | median canonical rank | tokens/query | guard candidate present | guard matched (replay) | pin changed window |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| status_quo | 0.78 | 0.90 | 0.54 | 3.00 (n=171/236) | 3832.8 | 1.00 | 0.00 | — |
| status_quo+pin | 0.78 | 0.90 | 0.54 | 3.00 (n=171/236) | 3832.8 | 1.00 | 0.00 | 0.00 |
| c_peers | 0.94 | 0.97 | 0.50 | 3.00 (n=154/236) | 1179.3 | 0.93 | 0.00 | — |
| c_peers+pin | 0.94 | 0.97 | 0.50 | 3.00 (n=154/236) | 1179.3 | 0.93 | 0.00 | 0.00 |
| b_grouped | 0.98 | 0.99 | 0.97 | 1.00 (n=232/236) | 1195.2 | 0.93 | 0.00 | — |
| b_grouped+pin | 0.98 | 0.99 | 0.97 | 1.00 (n=232/236) | 1201.5 | 0.93 | 0.00 | 0.06 |

Token counts come from the `char-proxy:4-chars-per-token` estimator (recorded because a substituted estimator would otherwise be indistinguishable from a measured one).  Guard threshold: 0.92.  Distractor slab: 300 records, identical in every arm.

## What the pin column shows

3111's topic anchor is ADDITIVE at the search seam (PRD D1): it appends a topic's canonical, it never promotes it over a ranked hit.  Both variants of a shape are scored over a window of the same size, so an additive pin can only pay off where a read-side transform freed a slot in that budget.  Per shape, as measured:

- **`status_quo`** — the pin changed 0.00 of the measured windows; every metric column is unchanged from `status_quo`.
- **`c_peers`** — the pin changed 0.00 of the measured windows; every metric column is unchanged from `c_peers`.
- **`b_grouped`** — the pin changed 0.06 of the measured windows; the `tokens_per_query` column(s) moved.

Read a `+pin` row against `pin changed window`, not against its twin alone.  A rate of `0.00` beside identical columns means the pin never fired — at a full window there is no slot for an appended record — which is a different finding from "the pin does not help".  A rate above zero beside identical columns means it fired and moved nothing these metrics measure.  Either way, a pin that is to pay off under a shape whose window is already full would have to PROMOTE rather than append, and that is a design choice for gate η, not a tuning knob for this experiment.

## D10 — audit-recall over the labeled fixture

Replay of `audit_duplicate_memories.find_near_duplicate_memory_groups` at threshold 0.85 over α/3130's curator-labeled fixture, scored against `calibrate_write_triage.build_pair_sets`.  **No threshold is asserted on this number** (gate G6): it informs how far to trust the κ duplicate sweep, it does not gate a build.

| class | pairs | recovered | rate |
| --- | --- | --- | --- |
| true duplicates | 301 | 0 | 0.00 |
| — lexical band (reachable) | 0 | 0 | — |
| — paraphrase band (unreachable) | 301 | 0 | 0.00 |
| hard negatives (falsely grouped) | 18 | 0 | 0.00 |
| unrelated (falsely grouped) | 5037 | 0 | 0.00 |

The **paraphrase band** is the positive pairs no character-level threshold can reach at any tuning short of changing kind (split by max `SequenceMatcher` ratio over both argument orders, since `ratio()` is order-sensitive).  Counting those as plain detector misses would read as "the audit script is broken" rather than "this class is invisible to it".  Nearest misses, for hand-auditing the split:

- `64564dd5-2c0d-4d0f-89e1-88d881da3d41` / `bf91bc5c-727c-48d8-a3cc-66e6938201ac` — max ratio 0.5464
- `76f848f3-39b6-4a1d-92d7-bca5e3de5dfe` / `de782198-1743-4efc-9915-8eee9cdc7e1c` — max ratio 0.4335
- `0b746438-6ce8-435c-885c-b3ac82666764` / `0f7b5fc7-fc11-4f5a-91f3-ab174e2216ed` — max ratio 0.4095
- `0e954870-bba1-44a7-81e9-5be94d5f6255` / `166f5106-00da-4364-9c6f-d0a0d514db7a` — max ratio 0.3794
- `1c2a0424-9e79-47f4-97f9-331160f5849a` / `2180e5df-7a95-475a-b123-165e55dd4770` — max ratio 0.3645

## Protocol

**Blind authoring**: single-author-blind-to-metrics, mechanized by commit ordering: the arm decomposition and query set were committed before any metric function existed in the tree.  The arm decomposition and query set were committed BEFORE any metric function existed in this tree; the commits below are the audit trail, and the anti-laziness floor is claim-coverage parity (every claim id realizable in every arm), deliberately not length parity — arm (a)'s long originals versus arm (c)'s short peers differ by construction, and that difference IS the tokens/query column.

**Embedder**: text-embedding-3-small.

| fixture | commit |
| --- | --- |
| `fused-memory/tests/fixtures/write_triage_calibration.jsonl` | 55e242a2c8681fb8a60fdb34e3d5194109781e87 |
| `fused-memory/tests/fixtures/memory_eval_topic_registry.json` | 02886ef290cde99ce88426cd1d3b5565a551e293 |
| `fused-memory/tests/fixtures/e2_arm_claims.jsonl` | f1fe9cf040dca324e3bd8cb4c13bf2e65da02620 |
| `fused-memory/tests/fixtures/e2_query_set.jsonl` | 9c48f3d10c2ee63aeb54443769903740d931aa4c |
| `fused-memory/tests/fixtures/e2_distractor_slab.jsonl` | 5c4a617265ae0de673825a5b2b881c45bef976ca |
