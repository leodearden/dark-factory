// memory_evals_fmt.js — pure display/vocabulary helpers for the memory-eval
// monitoring section rendered by tab_memory_evals.jsx.
//
// This is a plain-JS module: no JSX, no Babel. It is loaded two ways:
//   - In the browser, via a classic `<script src="/static/redux/memory_evals_fmt.js">`
//     tag (like spark_path.js/runtime_format.js), which assigns
//     `window.DF_MEMORY_EVALS_FMT`.
//   - In node (no package.json in this repo, so this file resolves as
//     CommonJS), via `require`/`import` for the `node --test` suite under
//     dashboard/tests/js/.
//
// Both export paths are guarded so this file has no effect outside the
// environment it's actually running in.
//
// index.html loads this file (classic script, before the Babel JSX tags) so
// `window.DF_MEMORY_EVALS_FMT` is defined before tab_memory_evals.jsx executes
// its top-level destructure of it.
//
// ── WHY THIS FILE EXISTS (task 3481) ───────────────────────────────────────
//
// Everything here is JSX-FREE branching logic that happened to be declared
// inside a `type="text/babel"` .jsx. That placement was the whole problem:
// tab_memory_evals.jsx is transformed by CDN Babel at runtime and this repo has
// no node_modules, so node cannot parse it and React cannot be rendered in any
// harness here. The only reachable test was therefore a Python suite that read
// the .jsx AS TEXT and asserted regexes over the source — an idiom that
//
//   1. never EXECUTES the logic (a regex can see that `verdictBadge` contains a
//      `+ ' · ' +` composition; it cannot see which of the 7 distinct verdict
//      inputs × 13 parity states actually produce which label);
//   2. passes SILENTLY when it matches nothing (two regexes in that suite were
//      found matching nothing at all, which is why it is full of `assert body`
//      anti-vacuity guards); and
//   3. drifted from a hand-pinned COPY of the parity vocabulary rather than the
//      real table — a three-member copy once went blind to six states.
//
// Moving these helpers into a plain-JS sibling puts the branching somewhere
// `node --test` can execute it with real assertions
// (dashboard/tests/js/memory_evals_fmt.test.mjs). tab_memory_evals.jsx keeps
// the JSX and destructures this module's API at its top level.
//
// What deliberately stays in Python: cross-language vocabulary completeness
// against the producer's frozenset `memory_evals.PARITY_STATES`, the PRD
// section 8 (G6/INV-5) source guard — which scans BOTH this file and the .jsx,
// since the logic it guards now lives here — index.html load order, and
// JSX/React render wiring.

// Missing scalars render an em-dash, never `|| 0`: a synthetic zero reads as a
// measured zero.  Same placeholder the Memory tab already uses (tabs.jsx:589).
function dash(v) {
  if (v === null || v === undefined) return '\u2014';
  return v;
}

// Compact age from `latest_run_age_seconds`.  Display only — the staleness
// THRESHOLD lives server-side and is deliberately absent from the payload, so
// nothing here can re-derive `stale`.
function ageText(seconds) {
  if (seconds === null || seconds === undefined) return '\u2014';
  const h = seconds / 3600;
  if (h < 1) return `${Math.round(seconds / 60)}m ago`;
  if (h < 48) return `${Math.round(h)}h ago`;
  return `${Math.round(h / 24)}d ago`;
}

// Count the deliberate holes in a trend series.  A `null` in `trend.values`
// means that run produced no sample.
//
// The array is handed on UNMODIFIED — dropping a hole would shift this
// metric's points against every other metric's, since all series share the
// run_stamps x-axis.
//
// HISTORY: charts.jsx's primitives originally could not REPRESENT a hole.
// `Sparkline` and `StepSpark` did their own arithmetic inline with no null
// handling, coercing a `null` to 0, so a hole was drawn as a real data point
// at value 0 connected by line segments to its neighbours — for a `proportion`
// metric sitting at 0.95, a plunge to the chart floor and back, visually
// identical to a genuine regression to zero.  That is fixed: the scale/path
// math now lives in /static/redux/spark_path.js (task 3436), which excludes
// non-finite samples from the extrema and breaks the path at every hole, and
// is behaviourally tested under `node --test`
// (dashboard/tests/js/spark_path.test.mjs).
//
// The local suppression below is nonetheless RETAINED for now: a holed series
// is NOT DRAWN (see `plottable` in tab_memory_evals.jsx) and the gap count is
// disclosed in text instead.  This is the same invariant `dash()` states for
// scalars — a synthetic zero reads as a measured zero — applied to the trend
// column.  Re-enabling this trend chart now that the primitive is hole-aware
// is a product decision with its own test churn
// (test_tab_memory_evals.py::test_trend_holes_are_never_handed_to_a_chart_primitive
// pins the current behaviour), and is filed as separate follow-up work.
function trendGaps(values) {
  if (!values) return 0;
  let gaps = 0;
  for (let i = 0; i < values.length; i++) {
    if (values[i] === null || values[i] === undefined) gaps += 1;
  }
  return gaps;
}

// ── Unmatched escalations ──
//
// Branched on `reason`, with distinct wording per value.  Collapsing the three
// into one undifferentiated "unexplained" list would fire on escalations that
// are in fact fully explained and train operators to ignore the one signal
// that catches a real parity orphan (memory_evals._unmatched_projection()).
function unmatchedReasonText(reason) {
  if (reason === 'no_matching_verdict') return 'no metric row explains this';
  if (reason === 'storm_suppressed') return "explained, but this run's links are collapsed into the aggregate";
  if (reason === 'no_fingerprint') return 'producer emitted no dedupe_fingerprint';
  return `unrecognised reason: ${String(reason)}`;
}

// ── Chart primitive per metric kind (PRD open question 1) ──
//
// RETURNS A TAG, NOT A COMPONENT.  The name is unchanged from the .jsx version
// but the return type is not, so a reader who assumes "component" is silently
// wrong.  This function used to return the `MEStep` / `MESpark` React component
// references destructured from `window.DF_CHARTS` — a browser global node
// cannot resolve, which is precisely what pinned this vocabulary inside a .jsx
// and out of reach of any runner (task 3481, DD-1).  It now returns a
// closed-vocabulary tag string and the tag→component mapping is the CALLER's
// job: tab_memory_evals.jsx holds a two-entry `ME_CHART_BY_TAG` lookup at its
// single call site.  That split is what lets node execute the vocabulary at all.
//
// The payload's kind vocabulary is exactly {tripwire, proportion, count,
// scalar}.  A kind outside that set is a RENDERING gap, not a data error: the
// builder passes the value through verbatim and files an `unknown_kind` issue
// for it.  So the fallback is `null` — value only, NO chart.  Guessing a
// primitive would render an unvalidated shape as though it were understood.
// The `null` tag also preserves the caller's `plottable` truthiness gate: an
// unknown kind resolves to a null Chart there exactly as it did before.
function chartForKind(kind) {
  if (kind === 'tripwire') return 'step';   // step-shaped item counts
  if (kind === 'proportion') return 'spark';
  if (kind === 'count') return 'spark';
  if (kind === 'scalar') return 'spark';
  return null;
}

// ── Verdict badge ──
//
// `verdict` and `parity` are the ONLY badge inputs.  Re-deriving alarm state
// from value-vs-limit in the browser is forbidden by PRD section 8 (G6/INV-5):
// memory_evals._parity() says `parity` exists precisely so the UI does not
// re-derive badge state "out of three separate fields, which is where the two
// sides would drift apart".  This function therefore performs string equality
// only — no arithmetic, no ordering comparison, no limits.
//
// The parity dimension REFINES the verdict; it never replaces it.
//
// ── The parity vocabulary, in ONE place ──
//
// These two declarations are this file's entire view of the parity vocabulary.
// Every member of memory_evals.PARITY_STATES must appear in EXACTLY ONE of
// them: PARITY_REFINEMENT for states whose badge carries a fact the verdict
// alone does not, PARITY_PLAIN for states where the verdict badge already says
// everything there is to say.  The plain list is an explicit opt-out, not an
// oversight bucket — "no branch for it" and "considered, nothing to add" are
// different decisions and only one of them is safe to leave undocumented.
//
// test_tab_memory_evals.py::test_parity_vocabulary_fully_covered asserts that
// split against the exported frozenset rather than a copy pinned in the test
// file, so a state added by the producer fails there instead of silently
// rendering through whatever the fall-through happens to be.  (A hand-picked
// three-member copy DID live in that test, and went blind to six states.)
//
// The values are SUFFIXES, never whole labels.  That is what makes "a parity
// branch may not discard the verdict" structural instead of re-checked per
// branch: there is no expression below capable of returning a label that omits
// `base`.
// Keys are QUOTED: they are the producer's vocabulary strings, looked up by
// string, not JS identifiers — and quoting keeps them greppable in the same
// form PARITY_PLAIN and the payload use.
// EVERY `_open` member renders the same 'escalation open' affordance, in the
// same words.  `_parity()` suffixes `_open` onto the linked variant of every
// non-alarm verdict class, so these all assert one fact — an escalation is
// filed and still live — and the operator should read one marker rather than
// learn six.  Without it the badge is identical to the same verdict with
// nothing filed, which is precisely the distinction parity exists to draw.
//
// Severity follows the VERDICT, not the linkage: an open escalation on a
// metric nothing judged alarming is a warn-level parity fact, not a healthy
// one and not an alarm.  `alarmed_open` keeps `badge bad` and pairs
// symmetrically with `alarmed_unlinked` — "alarm · escalation open" vs
// "alarm · no escalation" — where an unsuffixed alarm badge was ambiguous
// between "escalation linked" and "state nobody handled".
const PARITY_REFINEMENT = {
  'alarmed_open':           { suffix: 'escalation open', cls: 'badge bad' },
  'alarmed_unlinked':       { suffix: 'no escalation',   cls: 'badge bad' },
  'storm_collapsed':        { suffix: 'storm-collapsed', cls: 'badge bad' },
  'recovered_open':         { suffix: 'escalation open', cls: 'badge warn' },
  'insufficient_data_open': { suffix: 'escalation open', cls: 'badge warn' },
  'grandfathered_open':     { suffix: 'escalation open', cls: 'badge warn' },
  'unjudged_open':          { suffix: 'escalation open', cls: 'badge warn' },

  // The unknown-verdict pair is the one place severity does NOT follow the
  // verdict, because there is no readable verdict to follow.  The producer
  // names this condition as an issue kind — it files an `unknown_verdict`
  // issue naming the eval, metric and offending value — precisely so a value
  // outside the closed vocabulary fails toward "visibly unrenderable" rather
  // than toward the healthy label.  This is the last render step and must not
  // quietly undo that.
  //
  // The condition itself is named by the BASE, which reads 'unreadable
  // verdict' for a value that is present but outside the vocabulary (see
  // verdictBadge below).  So these two suffixes carry only what PARITY adds —
  // the linkage — exactly as the alarm pair does.  Naming the condition in
  // both halves would render "unreadable verdict · unrecognised verdict",
  // asserting one fact twice and reading as two.
  //
  // Borrowing `badge muted` here would report a PRESENT-but-unreadable verdict
  // as an ABSENT one — the same substitution the absent-verdict guard
  // prevents, running the other way.  `unjudged` is the genuinely-absent case
  // and correctly stays in PARITY_PLAIN, muted, reading 'no verdict'.
  'unknown_verdict':        { suffix: 'no escalation',   cls: 'badge bad' },
  'unknown_verdict_open':   { suffix: 'escalation open', cls: 'badge bad' },
};
// Reserved for states where the verdict badge already says everything there is
// to say: nothing is filed, and the verdict names itself.
//
// 9 refined + 4 plain = 13 = |memory_evals.PARITY_STATES|.
const PARITY_PLAIN = [
  'clear',
  'insufficient_data',
  'grandfathered',
  'unjudged',
];

// ── The verdict vocabulary, in ONE place ──
//
// Same discipline as the parity tables above, for the other axis: the four
// persisted verdicts and the base badge each earns, declared as data and
// EXPORTED, so both suites drive off the real vocabulary instead of a copy.
// Three copies of this list existed before task 3481's review — this table,
// a tuple in test_tab_memory_evals.py, and the producer's own
// `memory_evals._KNOWN_VERDICTS` — and only the producer's was authoritative.
// A fifth verdict added there would have rendered as 'unreadable verdict'
// with a `badge bad` while every test stayed green.
// test_tab_memory_evals.py::test_verdict_vocabulary_fully_covered now compares
// these keys against that frozenset, in both directions.
//
// Keys are QUOTED for the same reason PARITY_REFINEMENT's are: they are the
// producer's vocabulary strings, looked up by string and greppable in the form
// the payload carries.
//
// A verdict OUTSIDE this table is not defaulted into it — see verdictBadge's
// absent/unreadable split below, which is deliberately NOT table-driven
// because neither of those two states is a member of the vocabulary.
const VERDICT_BASES = {
  'alarm':             { base: 'alarm',             cls: 'badge bad' },
  'no_alarm':          { base: 'no_alarm',          cls: 'badge ok' },
  'grandfathered':     { base: 'grandfathered',     cls: 'badge info' },
  'insufficient_data': { base: 'insufficient_data', cls: 'badge muted' },
};
function verdictBadge(metric) {
  const verdict = metric.verdict;
  const parity = metric.parity;

  // The plain verdict label is computed FIRST, so that no parity branch below
  // can discard it.
  //
  // This ordering is load-bearing.  memory_evals._parity() is a THREE-case
  // lookup over (verdict class, linked?): `alarm`, then `no_alarm`, then a
  // per-class fall-through that suffixes `_open` onto the linked variant of
  // every remaining class.  So every verdict class carries its own pair of
  // states and the class survives into the parity string — `recovered_open`
  // and `clear` are NARROWED to the `no_alarm` verdict, and
  // `insufficient_data`, `grandfathered` and an absent verdict now reach
  // `insufficient_data_open`, `grandfathered_open` and `unjudged_open` rather
  // than borrowing the recovery label.
  //
  // The producer draws those distinctions precisely so this badge can keep
  // them, and a parity branch returning a FIXED label would collapse them
  // right back — telling the operator that a metric nothing judged had
  // recovered, the same "we did not measure" -> "we measured and it is fine"
  // substitution the absent-verdict fall-through exists to prevent.  Suffixing
  // `base` rather than replacing it is what makes that unrepresentable here,
  // for every one of the nine refined states at once.
  //
  // Absent is absent and unreadable is unreadable: neither is EVER defaulted
  // to no_alarm, and the two carry DIFFERENT labels — mirroring
  // memory_evals._verdict_class(), which buckets an absent value to `unjudged`
  // and a present-but-out-of-vocabulary one to `unknown_verdict`.
  //
  // The four in-vocabulary verdicts come from the VERDICT_BASES table; the two
  // states that are NOT verdicts — absent, and present-but-unreadable — are
  // branches here, because neither is a member of the vocabulary and putting
  // them in the table would make the completeness check against
  // `memory_evals._KNOWN_VERDICTS` report them as dead branches.
  //
  // `hasOwnProperty` rather than a bare `VERDICT_BASES[verdict]`, for the same
  // reason the parity lookup below uses it: a payload carrying verdict
  // 'constructor' / 'toString' would otherwise find a truthy INHERITED member
  // and render `base: undefined` — a value the operator reads as broken
  // tooling rather than as the unreadable verdict it actually is.
  let base = 'no verdict';
  let cls = 'badge muted';
  const knownVerdict = Object.prototype.hasOwnProperty.call(VERDICT_BASES, verdict)
    ? VERDICT_BASES[verdict]
    : null;
  if (knownVerdict) {
    base = knownVerdict.base;
    cls = knownVerdict.cls;
  } else if (verdict !== null && verdict !== undefined) {
    // Present, but outside the closed vocabulary — the bucket
    // memory_evals._verdict_class() calls `unknown_verdict`.  'no verdict' is
    // reserved for the genuinely ABSENT case: saying it here would assert the
    // verdict is MISSING, and the operator reads the rendered string, not this
    // comment.  `badge bad` rather than muted so the row fails toward
    // "something is wrong here" on the base alone, before any parity
    // refinement — the producer files a named `unknown_verdict` issue for
    // exactly this value and the last render step must not undo it.  Wording
    // follows `unmatchedReasonText`'s `unrecognised reason: ...`, so the file
    // has ONE register for "the producer emitted a value this UI does not
    // recognise".
    base = 'unreadable verdict';
    cls = 'badge bad';
  }

  // Parity states that carry information the verdict alone does not.  Each
  // REFINES the base label by suffixing it; none replaces it.  `alarmed_unlinked`
  // and `storm_collapsed` are only reachable when verdict === 'alarm', so those
  // read as "alarm · ..." as before — composing rather than hard-coding just
  // keeps them honest if the producer's derivation ever widens.
  //
  // One lookup and one composition site.  A state listed in PARITY_PLAIN is not
  // absent from the table by accident; it is declared there as having nothing
  // to add to the verdict badge.
  // `hasOwnProperty` rather than a bare `PARITY_REFINEMENT[parity]`: the table
  // is a plain object literal, so a bare lookup resolves through
  // Object.prototype and a payload carrying parity 'constructor' / 'toString'
  // / 'valueOf' / 'hasOwnProperty' would find a truthy INHERITED member,
  // rendering `cls: undefined` and a label ending in '· undefined'.
  const own = Object.prototype.hasOwnProperty.call(PARITY_REFINEMENT, parity);
  const refinement = own ? PARITY_REFINEMENT[parity] : null;
  if (refinement) {
    return { cls: refinement.cls, label: base + ' · ' + refinement.suffix };
  }

  // A non-empty parity in NEITHER declaration is a state this copy of the file
  // has never been told about — the producer added one and an already-open
  // browser is still holding a cached bundle.  It is MARKED, not passed
  // through: rendering it as the bare verdict badge is indistinguishable from
  // a state that declined refinement, i.e. the unknown fails toward the
  // healthy label — the same substitution the unknown-verdict pair above
  // exists to prevent, one level up.  PARITY_PLAIN is what makes "considered
  // and declined" distinguishable from "never heard of it" at all.
  if (parity && PARITY_PLAIN.indexOf(parity) === -1) {
    return { cls: 'badge bad', label: base + ' · unrecognised parity' };
  }
  return { cls: cls, label: base };
}

const MEMORY_EVALS_FMT_API = {
  dash,
  ageText,
  trendGaps,
  unmatchedReasonText,
  chartForKind,
  verdictBadge,
  // BOTH badge vocabularies are EXPORTED, not merely used: the node suite
  // enumerates its verdict×parity matrix off these tables so a state the
  // producer adds is covered the moment it lands here, rather than off a
  // hand-pinned copy that can go blind to states nobody remembered to add.
  PARITY_REFINEMENT,
  PARITY_PLAIN,
  VERDICT_BASES,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = MEMORY_EVALS_FMT_API;
}
if (typeof window !== 'undefined') {
  window.DF_MEMORY_EVALS_FMT = MEMORY_EVALS_FMT_API;
}
