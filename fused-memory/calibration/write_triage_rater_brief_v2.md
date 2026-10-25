# Rater brief v2.1: what a reader of the record would miss

This brief replaces the seven-word attach scale (`write_triage_rater_brief.md`,
kept unchanged as the record of v1) for every rating under rubric v2.1. It asks
three questions about a pair of records, and follow-ups for two of them. The
answers are not verdicts about where a write belongs. They are facts about the
two texts, and the scorer turns them into correctness mechanically.

## The item

Each item is a blinded pair: CHILD, a write, and PARENT, an earlier record. You
are not told what the system did with CHILD, which system produced the pair,
or anything from a key file. Judge the two texts only.

Texts are capped at 4,000 characters. A cut text ends with the marker
`…[truncated, N chars total]`. PARENT is shown as it reads NOW. A shard line
saying `basis=attach_time` instead shows PARENT as it read when CHILD was
written; judge that text the same way.

You are not asked what a reader of PARENT sees of CHILD: the scorer computes
that from your `loss_spans`
(`fused-memory/scripts/write_triage_reader_rubric.py::digest_visible`).

## The questions

**HOME.** Is CHILD's central claim PARENT's? It is when CHILD's central claim
is the same assertion about the same specific named thing as PARENT's: a
function, a test, a procedure, a command, a defect, a decision, a measurement
series, a task. Same subject with a different claim is NO. This holds for
every record, including orchestrator completion records: judge the claim,
never the task id. Details, caveats and newer measurements that CHILD adds
never make HOME NO; they are what CONFLICT and LOSS ask about.

**CONFLICT.** Does CHILD show that something PARENT states is false, outdated
or replaced? The statement may be PARENT's central claim or a secondary one:
a status, a count, a measurement, an "X does not exist", or a recommendation
that the state CHILD describes makes untrue. CHILD need not mention PARENT.
A "Completed:" record asserts that its change landed: the defect it describes
is the state before the change, so it disputes a PARENT that still states the
defect as current. An exception disputes PARENT only when PARENT says always,
only or never and the exception falls inside PARENT's scope; any other
exception is a caveat, which LOSS asks about. CONFLICT tests truth alone,
never whether an agent would act differently.

When CONFLICT is YES, also give:

- **SCOPE.** Is the disputed statement PARENT's central claim or a
  load-bearing part of it (a rule, command, threshold, mechanism or
  decision)? Then CENTRAL. Is it a secondary detail? Then SECONDARY. SCOPE
  records how much of PARENT is in dispute. It never makes a dispute wrong.
- **action_changing.** Would an agent believing CHILD act differently from one
  believing PARENT? YES or NO. It is recorded for the report and never decides
  correctness.
- **conflict_quote.** `{"parent": ..., "child": ...}`: the PARENT sentence
  that CHILD shows false, outdated or replaced, and the CHILD sentence that
  shows it, each copied verbatim from the text shown.

**LOSS** (only when HOME is YES and CONFLICT is NO). Would someone acting on
PARENT's claim go wrong, or lack something they need, without CHILD?
Provenance (who, when, which branch) does not count, and neither does bare
recurrence (the same thing seen again). When LOSS is YES, give `loss_spans`:
verbatim quotes from CHILD of exactly what is needed, one entry per separate
piece.

Answer HOME, CONFLICT, LOSS and action_changing with YES or NO, and SCOPE with
CENTRAL or SECONDARY. UNCLEAR is allowed only when a truncation falls at the
point that decides the answer, and then `unclear_reason` must name that point.

Quotes and spans are checked against the texts shown. Differences in
whitespace and line breaks are ignored, and nothing else is. A quote or span
that is not found is sent back to you to answer again.

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
   duration over more tests, with the same timeout advice. CHILD's central
   claim, the run's duration and the advice, is PARENT's, so HOME is YES. The
   new number shows that PARENT's measurement is outdated, so CONFLICT is YES.
   The dispute is about a number, while the advice stands, so SCOPE is
   SECONDARY. An agent would follow the same advice either way, so
   action_changing is NO. `conflict_quote` gives PARENT's duration sentence and
   CHILD's.
2. **A dispute of a secondary point.** PARENT is a recipe for proving that a
   test-file split moves tests faithfully, and in passing names which linter
   binary to use. CHILD is about measuring one lint rule's count and says that
   the linter is now pinned in the project. CHILD's claim is not PARENT's, so
   HOME is NO. CHILD shows that the binary PARENT names is no longer the one in
   use, so CONFLICT is YES and SCOPE is SECONDARY.
3. **A restatement that adds one detail.** PARENT says a write check only
   recognises tasks spelled "task N", and that declaring a bare id is
   rejected. CHILD says the same and adds that the rejected write stored
   nothing, so retrying it after rewording is safe. HOME is YES, and CHILD
   shows nothing PARENT states to be false, so CONFLICT is NO. LOSS is YES:
   someone retrying a rejected write on PARENT's word would not know that the
   retry is safe. `loss_spans` quotes CHILD's retry-safety sentence.
4. **Two completion records.** PARENT and CHILD are both templated completion
   records from the same project's census work. Each one reports a different
   task's change: PARENT the dedup protocol, CHILD the feedback of task
   outcomes. The template and the subject area are shared, but the claims are
   not, so HOME is NO.

## Procedure

- Read items in small batches. Read each text in full, up to its length, and
  never judge from a short prefix. Judge each item independently.
- Write one JSON line per item:
  `{"id", "home", "conflict", "scope"?, "action_changing"?, "conflict_quote"?, "loss"?, "loss_spans"?, "unclear_reason"?, "reason"}`.
  Give `scope`, `action_changing` and `conflict_quote` when, and only when,
  `conflict` is YES. Give `loss` when, and only when, `home` is YES and
  `conflict` is NO, and `loss_spans` when, and only when, `loss` is YES.
  `reason` is one sentence that names both claims.
- Every item gets two blind raters. A third rater is asked only where the two
  disagree on a binary that decides the item's correctness.
