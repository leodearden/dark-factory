# Rater brief v2.2: what a reader of the record would miss

v2.2 (2026-10-29): the LOSS question is rewritten as one test with a list of what does not count, and anchors 5 to 8 are added; HOME, CONFLICT, SCOPE, action_changing, the output format and the procedure are unchanged from v2.1.

This brief replaces the seven-word attach scale (`write_triage_rater_brief.md`,
kept unchanged as the record of v1) for every rating under rubric v2.2. It asks
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

**LOSS** (only when HOME is YES and CONFLICT is NO). Does CHILD add something
that a reader relying on PARENT would need? Take each thing CHILD says that
PARENT does not already say or make obvious, and put it to one test: *a reader
who does exactly what PARENT says, or acts on what PARENT states, in the
situation PARENT describes, knowing nothing of CHILD: would they act wrongly,
or be unable to finish, without this?* LOSS is YES when at least one thing
passes the test, and NO otherwise. Three kinds of thing pass it:

- a case, condition or exception in which PARENT's claim or instruction does
  not hold as stated, or needs a further step: the caveat that CONFLICT leaves
  to LOSS;
- a consequence of PARENT's claim that its reader would not otherwise expect
  and would act on wrongly;
- a step, value, command, format or location that PARENT's instruction depends
  on and does not give, so that the instruction cannot be carried out from
  PARENT alone; or a remedy or workaround for a hazard PARENT names but gives
  no remedy for.

These never pass it on their own, and a CHILD that adds only them is LOSS NO:

1. Another route to an end PARENT already gives a route to: a different
   command, check, filter, detection or remedy, or PARENT's mechanism in
   other or more exact words.
   PARENT's own route is taken to work; a CHILD that shows it fails is
   CONFLICT, not LOSS.
2. PARENT's rule applied to one more case, stated for more cases, or shown by
   more examples, when PARENT's rule as given already tells its reader what
   to do.
3. The same thing seen again: a further sighting, measurement, count, date or
   size, or evidence that PARENT's claim holds in more places.
4. Provenance and backing for CHILD's own claim or for the change it records:
   who, when, which branch, task, commit or ticket; CHILD's reasons and
   premises, its plans, line anchors, the commands CHILD ran to check itself,
   and the pre-change state that a completion record describes. A command or
   step PARENT's instruction needs is the third kind that passes, and a
   consequence that makes some other statement false is the second, not
   backing.
5. A fact about something else that stands next to PARENT's subject: a tip, a
   related gotcha, a sibling task.
6. Confirmation that a change PARENT already describes has landed, or that
   PARENT still holds.

When LOSS is YES, give `loss_spans`: verbatim quotes from CHILD of exactly
what passes the test, one entry per separate piece, and nothing from the list
above.

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

Anchors 1 to 4 are paraphrased from hard cases in the 2026-10-19 soak. Anchors
5 to 8 are synthetic: each describes a shape, not a pair.

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
5. **Another route.** PARENT says to confirm that a service has finished
   starting by polling its health endpoint, and only then to send it work.
   CHILD says to watch the service's log for its ready line instead, and then
   to send it work. HOME is YES and CONFLICT is NO. LOSS is NO: PARENT's
   reader reaches the same point, and the same action, by PARENT's own check.
   Had CHILD shown that the health endpoint answers before the service is
   ready, that would be CONFLICT.
6. **A wider statement.** PARENT says a config loader accepts a duration only
   as a bare number of seconds, and tells its reader to write seconds. CHILD
   adds that `30m` and `2h` are rejected as well, with the same advice. HOME
   is YES and CONFLICT is NO. LOSS is NO: PARENT's advice already keeps its
   reader right in every case CHILD lists. Had PARENT said only that `30m` is
   rejected, without the bare-seconds rule, CHILD's statement that every unit
   suffix is rejected would pass the test: PARENT's reader writing `2h` would
   act wrongly.
7. **A completion record under a record of the landed state.** PARENT
   describes how a mechanism works now and names the task that changed it.
   CHILD is that task's completion record: the defect as it was, the premises
   verified, the plan, the key decisions and the modules touched. HOME is
   YES. CONFLICT is NO, since PARENT does not state the defect as current.
   LOSS is NO unless the record also states a case, condition or consequence
   that passes the test: the defect as it was, the premises, the plan, the
   rationale and the fact of landing do not.
8. **A missing step.** PARENT says a release note is generated from commit
   trailers and must never be edited by hand. CHILD says the same and gives
   the trailer's required form and the command that regenerates the note.
   HOME is YES and CONFLICT is NO. LOSS is YES: without the form and the
   command, PARENT's reader cannot carry out PARENT's instruction.
   `loss_spans` quotes both.

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
