# `normalize_topic_slugs.py` — Operator Guide

Corpus-wide `metadata.topic` slug normalization (task 4878). The script folds
every non-conforming live `topic` value into the shape `fused_memory.topic_slug`
defines, and moves the `x_recon_consolidation_gate` block of any consolidation
gate filed against a renamed slug in lockstep with its memories.

Why it is a corpus sweep, and its scope boundary, are in the script's module
docstring. How a slug moves through both stores, and why it moves
all-or-nothing, is in the docstring of
`fused-memory/src/fused_memory/maintenance/topic_slug_lockstep.py`.

Dry run is the default.

---

## Step 1 — Dry run

From anywhere, naming the checkout of EACH swept corpus:

```bash
uv run --project fused-memory python \
    fused-memory/scripts/normalize_topic_slugs.py \
    --project-root <dark-factory checkout> \
    --project-root <reify checkout>
```

Writes `plans/topic-slug-normalization-report.{json,md}` and prints the
markdown. Nothing is modified. `--project-root` REPLACES the default (this
checkout alone), so a swept corpus whose checkout is not named is a
`gate_census_incomplete` gap: the dry run exits 1 and names it.
`--topic <legacy value>` (repeatable) narrows the write set to the named
legacy spellings; it is how one registry topic is normalized without sweeping
the whole project.

---

## Step 2 — Read the refusal buckets and resolve them BY HAND

`slug_collision`, `canonical_collision`, `topic_unfoldable` and
`gate_ambiguous` are the buckets this script deliberately will not decide for
you: the first two would merge two topic namespaces (and, in the canonical
case, manufacture the second canonical `_check_canonical_uniqueness` exists to
prevent), the third has no honest fold, so any value the script picked would
be a guess written across the whole corpus, and the fourth is a slug carrying
more than one consolidation gate, where moving one gate would strand the
others. Also check `under_enumerated` / `scroll_budget_exhausted`: a run with
either populated is a LOWER BOUND, and a clean residue probe over it proves
nothing.

After an `--apply`, `ungated_partial_rename` lists any slug rolled back
because a member's write failed: resolve that failure and re-run. An entry
with non-empty `on_new_slug_memory_ids` is still split across both values, and
a re-run refuses it as `slug_collision` until it is put back by hand: patch
each member named in `on_new_slug_memory_ids` back to the legacy value with a
metadata-only `update_memory` merge, the same patch this script issues, then
re-run once the straggler's failure is resolved.

---

## Step 3 — Apply

As an OPERATOR, with the same roots:

```bash
... normalize_topic_slugs.py --apply \
    --project-root <dark-factory checkout> \
    --project-root <reify checkout>
```

`--apply` refuses before scrolling anything (`GateCensusIncomplete`) while any
swept project is uncovered or any named root failed to answer. This is an
operator action, not an agent one. The store-mutation preflight fails closed
for any process that cannot write mem0's history directory, which is the
normal posture inside an agent sandbox — so a dry run is what an agent can
deliver, and the deliverable of task 4878 is THIS INSTRUMENT PLUS A VERIFIED
DRY RUN, not the live mutation.

---

## Step 4 — Re-measure

With the standing instrument, and compare against the committed baseline:

```bash
... census_memory_metadata.py   # read coverage.topic_coverage.slug_non_conforming
```

The report's `baseline_delta` block already carries the arithmetic; step 4 is
how you confirm it against the corpus rather than against this run's own
enumeration.
