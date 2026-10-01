# Read-transform selection over the ratified C write shape

**Write shape:** `c_peers` (ratified by E2 gate ζ). This document does not re-litigate the write shape; it chooses the READ transform layered on top of it.
**Token estimator:** `char-proxy:4-chars-per-token` — the same one the committed E2 table resolved, so the token columns are directly comparable across the two artifacts.
**Windows:** authored set at k=10; production set at k=5 **by choice**. The four briefing-assembler queries — the modal read — do fire at `limit=5` (`orchestrator/src/orchestrator/agents/briefing.py`:1376), and that line governs nothing else: the sampled residual tail comes from other callers, measured in the journal at limits from 3 to 50. The tail is scored at the briefing's k for comparability, not because that k was observed on it; each row carries its own measured `observed_limits` histogram in the sample and the JSON artifact.

## The decision table

A `—` cell is **no measurement**, never a measured zero.

| arm | claim recall | canonical in top-k | tokens/query | topic diversity | baseline retention | displacement | prod tokens/query | prod displacement | drops ranked records | needs `contested` for V2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flat read (baseline) | 0.97 | 0.65 (aliased) / 0.65 (unaliased) | 1181.29 | 2.72 | 1.00 | 0.00 | 1240.48 | 0.00 | no | no |
| promoting topic pin | 1.00 | 1.00 (aliased) / 0.65 (unaliased) | 1070.27 | 2.72 | 0.83 | 1.71 | 1203.84 | 0.32 | no | no |
| topic-keyed grouped read | 1.00 | 1.00 (aliased) / 0.65 (unaliased) | 1308.37 | 2.72 | 1.00 | 6.43 | 1261.43 | 0.39 | yes | yes |
| topic-diversity cap | 0.63 | 0.65 (aliased) / 0.65 (unaliased) | 790.23 | 2.72 | 0.53 | 4.72 | 1234.77 | 0.07 | yes | no |

The production half carries no claim-recall or canonical column at all: those cells would be `—` for every arm. Production queries carry no ground truth: the reconciliation write journal records which query was issued, never which memory should have been returned. Claim recall and canonical discoverability are therefore not computable over this set and render as no-measurement, not as a measured zero.

## Disclosure (a): the `record_id` aliasing

A grouped read emits its document under the **canonical's own** `record_id` (`bake_off_storage_shape.py`:934-935). Any metric that credits the canonical on `hit.record_id == canonical_record_id` therefore scores a fold as "canonical found" — whether or not the canonical's own stored record ever ranked. That is a property of the TRANSFORM, not of retrieval.

This is the mechanism behind the committed E2 table's `b_grouped` canonical-in-top-5 of 0.97: it is an aliased rate, and reading it as "retrieval finds the canonical 97% of the time" is not what was measured. Both rates are printed above, in one cell each, so neither can be quoted without its semantics.

* **aliased** — the legacy `record_id`-match semantics over the DELIVERED window, preserved so this table stays comparable with the committed one. A PLACEMENT property: what the reader got under the canonical's id.
* **unaliased** — a RETRIEVAL property: the canonical's OWN stored record ranked, and where. Computed from the untransformed, untruncated hit list, so no read transform can touch it — which makes it **identical across every arm in the table by construction**, and that identity is the finding, not an error. A column that varied by arm here would be measuring the transform again.

Because it is arm-invariant, the unaliased column cannot discriminate between arms and is therefore **not** a sort key in the selection rule below; neither is the aliased one, which would reward minting a document under the canonical's id. See the Recommendation.

## Disclosure (b): the sighting-crediting knob

**Setting for this run: `uncredited`.**

In the landed grouped read this is not a knob at all but a hard-coded policy (`bake_off_storage_shape.py`:927): a sighting is collapsed to a bare count, its body is not rendered, and its claims are not credited. Arm (2) exposes it as a dial.

The consequence is arithmetic, not a verdict on any transform: with sightings uncredited the claim-recall **ceiling** is exactly `(claims - sightings + contested) / claims`. An arm cannot exceed that ceiling however well it retrieves, so a recall column read without the knob setting beside it is not interpretable.

## Disclosure (c): which arms suppress

"Suppressing read" is **two independent facts**, kept in two columns above because collapsing them would mislead task 3111:

* **flat read (baseline)** — drops no ranked record; needs no `contested` key.
* **promoting topic pin** — drops no ranked record; displaces the k-th record at an already-full window; needs no `contested` key.
* **topic-keyed grouped read** — drops ranked records from the window; would need a `contested` key for PRD V2.
* **topic-diversity cap** — drops ranked records from the window; needs no `contested` key.

An arm can suppress and still be landable today (the topic-diversity cap drops records, yet computes its cap from `metadata['topic']` alone and never asks whether a record is contested). Reporting one merged "suppressing" boolean would have said the cap is blocked on a key that does not exist.

## Disclosure (d): PRD V2's protection is unimplementable today

PRD V2 requires that **contested children are never suppressed** — the esc-5712 protection. That protection cannot be implemented at all right now, for any arm, and this is a precondition on task 3111 rather than a property of any transform measured here.

`contested` is a hand-labelled **fixture** field of the bake-off (`ArmClaim.contested`:196, `contested_record_ids`:2567, `SeededArm.contested_ids`:2593). It never appears in any `ArmRecord.metadata`, it has **no writer**, and it has no adjudication surface anywhere in the running system.

The live reserved vocabulary is `RESERVED_VOCABULARY_KEYS` (`fused-memory/src/fused_memory/memory_metadata.py::RESERVED_VOCABULARY_KEYS`), and it is exactly {`canonical`, `kind`, `parent_id`, `supersedes`, `topic`} — verified against the imported frozenset, not transcribed. `contested` is not among them. Any arm that suppresses records therefore ships without the esc-5712 protection until a `contested` key is designed, reserved, written and adjudicated.

## Recommendation

**Task 3111 should land: promoting topic pin.**

*Selection rule (fixed before the numbers were read):* Landability first: any arm that would need a `contested` key to satisfy PRD V2 is excluded outright, because that key does not exist. Among the rest, rank by claim recall, then tokens/query (lower better), then window displacement (lower better). NEITHER canonical column is a sort key: the UNALIASED one is arm-invariant by construction — it is read from the pre-transform fetch, so no read transform can move it, and ranking on it would sort every arm on the same constant — while the ALIASED one is a placement property, so ranking on it would reward minting a document under the canonical id. Both are reported in full and neither decides. A None sorts last, never as a zero. The rule is fixed before the numbers are read and is not re-tuned to move a column.

promoting topic pin is the highest-ranked arm that task 3111 can actually land today: it satisfies the ordering above while requiring no vocabulary key that has no writer. Arms excluded pending a `contested` key: ['topic-keyed grouped read'].

**No arm improves the canonical's retrieval rank.** Every arm in the table scores an unaliased rank of 3.49 — the flat read's own value — because that column is read from the untransformed, untruncated fetch, which no read transform touches. What the three transforms move is PRESENTATION: the aliased rank goes to flat read (baseline) 3.49, promoting topic pin 1.06, topic-keyed grouped read 1.11, topic-diversity cap 1.05, i.e. what the reader gets under the canonical's id, while the canonical's own stored record still ranks exactly where the store put it. Task 3111 is therefore choosing how the window is ARRANGED, not how well it is retrieved; a change that needs the canonical to be retrieved better cannot be made at this layer at all.

Excluded outright, at any score: **topic-keyed grouped read** — see disclosure (d). This is an exclusion on landability, not a measurement verdict; the arm's row above is still reported in full so a later reader with a `contested` key can re-decide.

## The production query set

Sampled READ-ONLY from the reconciliation write journal (`fused-memory/scripts/harvest_production_queries.py`, `mode=ro` + `PRAGMA query_only`). The four briefing-assembler templates below are the high-volume core; the shares are measured counts over that journal, not estimates.

| query template | match | observed | share of all search traffic |
| --- | --- | --- | --- |
| `project overview architecture goals` | literal | 74,702 | 17.31% |
| `coding conventions and project norms` | literal | 74,944 | 17.36% |
| `recent decisions and rationale` | literal | 74,884 | 17.35% |
| `task {task_id} context and related decisions` | parameterized | 54,097 | 12.53% |

Together the four templates are **64.55%** of all search traffic in the journal, so an arm's cost on them is not a corner case — it is the modal read. The rest of the sample is a bounded deterministic tail draw from the long tail of one-off queries.

### The scoring window is a choice, not a reading

**These four templates** fire at `limit=5` in production (`briefing.py`:1376). That line governs the briefing assembler and nothing else, so it is not evidence about any other query.

* briefing family, measured: `3`×1, `5`×278,617, `6`×1, `10`×4, `20`×4
* sampled tail, measured: `3`×2, `4`×2, `5`×104, `6`×19, `8`×36, `10`×599, `15`×194, `20`×75, `30`×11, `50`×99

The production half of the table above is scored at k=5 for **every** row, tail included. For the tail that is a CHOICE — made so the production columns are comparable with each other and with the authored set — and not a limit observed on those queries. The per-row readings are in `observed_limits` in both the sample fixture and the JSON artifact; a reader who wants each row scored at its own k has the numbers to ask for it.

## Why the production columns carry no labels

The production query set is sampled read-only from the reconciliation write journal (`fused-memory/scripts/harvest_production_queries.py`). It is **unlabeled by construction**: the journal records which query was issued, never which memory should have come back. Claim recall and canonical discoverability are not computable over it, so they are reported as no-measurement rather than as a zero, and no arm is credited or penalised for them on this set. Inventing a label here would fabricate the very ground truth the measurement exists to establish.
