# Rater brief v2: what a reader of the record would miss

This brief replaces the seven-word attach scale (`write_triage_rater_brief.md`,
kept unchanged as the record of v1) for every rating under rubric v2. It asks
four questions about a pair of records, plus a follow-up for two of them. The
answers are not verdicts about where a write belongs. They are facts about the
two texts, and the scorer turns them into correctness mechanically.

## The item

Each item is a blinded pair: CHILD, a write, and PARENT, an earlier record. You
are not told what the system did with CHILD, which system produced the pair,
or anything from a key file. Judge the two texts only.

Texts are capped at 4,000 characters. A cut text ends with the marker
`…[truncated, N chars total]`. PARENT is shown as it reads NOW. A shard line
saying `basis: attach_time` instead shows PARENT as it read when CHILD was
written; judge that text the same way.

Each item also shows a CHILD DIGEST: the first 240 characters of CHILD, which
is all a reader of PARENT sees of an amendment. You need it only for DIGEST.

## The four questions

**HOME.** Does PARENT state the same claim as CHILD? The same claim is the same
assertion about the same specific named thing: a function, a test, a
procedure, a command, a defect, a decision, a measurement series, a task. Same
subject with a different claim is NO. This holds for every record, including
orchestrator completion records: judge the claim, never the task id. A CHILD
that makes several load-bearing claims is YES only if every one of them is
PARENT's.

**LOSS.** Does CHILD hold a fact, number, condition, cause, fix, caveat or newer
state that a reader acting on PARENT alone would need? Provenance (who, when,
which branch) does not count, and neither does bare recurrence (the same thing
seen again).

**CONFLICT.** Does CHILD say that something PARENT states is false, outdated or
different now, such that an agent believing CHILD would act differently? The
statement may be PARENT's central claim or a secondary one. A newer state that
replaces PARENT's claim, such as a supersession or a measurement update, is a
CONFLICT.

**SCOPE** (only when CONFLICT is YES). Is the disputed statement PARENT's
central claim or a load-bearing part of it (a rule, command, threshold,
mechanism or decision)? Then CENTRAL. Is it a secondary detail? Then SECONDARY.
SCOPE records how much of PARENT is in dispute. It never makes a dispute wrong.

**DIGEST** (only when LOSS is YES). Is everything LOSS points to visible in the
CHILD DIGEST shown? Answer YES or NO.

Answer HOME, LOSS and CONFLICT with YES or NO, and SCOPE with CENTRAL or
SECONDARY. UNCLEAR is allowed only when a truncation falls at the point that
decides the answer, and then `unclear_reason` must name that point.

## How answers become correctness

You do not decide whether the system was right; the scorer does, from your
answers. The rules are implemented once, in
`fused-memory/scripts/write_triage_reader_rubric.py::derive`, and the metrics
built on them are `fused-memory/calibration/write_triage_reader_metrics.md`.
In short, an outcome is correct when:

| Outcome | Correct when |
|---|---|
| restated | HOME and not LOSS and not CONFLICT |
| amended | HOME and not CONFLICT |
| contested | CONFLICT |
| stored | there is no earlier record it should have joined |

## Anchors

Paraphrased from hard cases in the 2026-10-19 soak.

1. **Measurement update.** PARENT records a verify run's duration and advises an
   explicit foreground timeout. CHILD, from a later run, records a shorter
   duration over more tests, with the same timeout advice. The new number
   replaces PARENT's measurement, so CONFLICT is YES. The dispute is about a
   number, while the advice stands, so SCOPE is SECONDARY.
2. **A dispute of a secondary point.** PARENT is a recipe for proving that a
   test-file split moves tests faithfully, and in passing names which linter
   binary to use. CHILD is about measuring one lint rule's count and says that
   the linter is now pinned in the project, so the binary PARENT names is no
   longer the one to use. CHILD's claim is not PARENT's, so HOME is NO. CHILD
   still disputes something PARENT states, and an agent would act on it, so
   CONFLICT is YES and SCOPE is SECONDARY.
3. **A restatement that adds one detail.** PARENT says a write check only
   recognises tasks spelled "task N", and that declaring a bare id is
   rejected. CHILD says the same and adds that the rejected write stored
   nothing, so retrying it after rewording is safe. HOME is YES and LOSS is
   YES, since the retry fact is something a reader of PARENT would need.
4. **Two completion records.** PARENT and CHILD are both templated completion
   records from the same project's census work. Each one reports a different
   task's change: PARENT the dedup protocol, CHILD the feedback of task
   outcomes. The template and the subject area are shared, but the claims are
   not, so HOME is NO.

## Procedure

- Read items in small batches. Read each text in full, up to its length, and
  never judge from a short prefix. Judge each item independently.
- Write one JSON line per item:
  `{"id", "home", "loss", "conflict", "scope"?, "digest"?, "unclear_reason"?, "reason"}`.
  Give `scope` only when `conflict` is YES and `digest` only when `loss` is
  YES. `reason` is one sentence that names both claims.
- A few blind anchors, items whose answers are already known, are mixed into
  each shard to detect drift between batches.
- Every item gets two blind raters. A third rater is asked only where the two
  disagree on a binary that decides the item's correctness.
