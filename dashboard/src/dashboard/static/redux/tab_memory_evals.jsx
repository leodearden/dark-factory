/* tab_memory_evals.jsx — memory-eval monitoring section, rendered INSIDE the
   Memory tab (PRD DD3: not a thirteenth top-level tab, no Rail entry).

   No JS test runner in this project (see scheduler_drawer.jsx comment).
   Wiring contracts are verified via Python source-assertion tests in
   dashboard/tests/test_tab_memory_evals.py.

   LOAD ORDER IS THE CONTRACT — this file must load BEFORE tabs.jsx.
   tabs.jsx destructures window.DF_MEMORY_EVALS at module top level (the
   tab_scheduler.jsx:15 / window.DF_SCHED_HEATMAP precedent), so a later tag
   would leave the global undefined at tabs.jsx evaluation time, throw, and
   blank every tab defined in that file.  Note this is the OPPOSITE direction
   from tab_escalations.jsx, which loads AFTER tabs.jsx because it additively
   mutates the window.DF_TABS object tabs.jsx creates.

   The payload (DF_DATA.MEMORY_EVALS) is produced by
   dashboard/data/redux_api.py::shape_memory_evals — those field names are the
   contract.  This file CONSUMES the server's judgments; it never re-derives
   them.  In particular nothing here compares a metric value against a limit.

   Exports: window.DF_MEMORY_EVALS = { MemoryEvalsSection, chartForKind, verdictBadge }
*/
const { Sparkline: MESpark, StepSpark: MEStep, PALETTE: MEC } = window.DF_CHARTS;
const MEDF = window.DF_DATA;

// ── Chart primitive per metric kind (PRD open question 1) ──
//
// The payload's kind vocabulary is exactly {tripwire, proportion, count,
// scalar}.  A kind outside that set is a RENDERING gap, not a data error: the
// builder passes the value through verbatim and files an `unknown_kind` issue
// for it.  So the fallback is `null` — value only, NO chart.  Guessing a
// primitive would render an unvalidated shape as though it were understood.
function chartForKind(kind) {
  if (kind === 'tripwire') return MEStep;   // step-shaped item counts
  if (kind === 'proportion') return MESpark;
  if (kind === 'count') return MESpark;
  if (kind === 'scalar') return MESpark;
  return null;
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
// is NOT DRAWN (see `plottable`) and the gap count is disclosed in text
// instead.  This is the same invariant `dash()` states for scalars — a
// synthetic zero reads as a measured zero — applied to the trend column.
// Re-enabling this trend chart now that the primitive is hole-aware is a
// product decision with its own test churn
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
  let base = 'no verdict';
  let cls = 'badge muted';
  if (verdict === 'alarm') {
    base = 'alarm';
    cls = 'badge bad';
  } else if (verdict === 'no_alarm') {
    base = 'no_alarm';
    cls = 'badge ok';
  } else if (verdict === 'grandfathered') {
    base = 'grandfathered';
    cls = 'badge info';
  } else if (verdict === 'insufficient_data') {
    base = 'insufficient_data';
    cls = 'badge muted';
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

// ── Escalation link ──
//
// A real <button>, not an <a href="#esc/...">.  The SPA has no router and no
// anchors anywhere in static/redux — tab state is React state in app.jsx and
// rows are selected with onClick handlers — so a fragment href would be a dead
// affordance.  `onNavigate` is threaded down from app.jsx (step-18); when it is
// absent the control renders disabled with a title saying so, never silently
// inert.
//
// Built from `escalation.id` alone: the projection carries exactly id, summary,
// severity, level, created_at and dedupe_fingerprint — there is no url.
function EscalationLink({ escalation, onNavigate }) {
  const wired = !!onNavigate;
  return (
    <button
      type="button"
      className="chip esc-link"
      data-testid="memory-eval-escalation-link"
      disabled={!wired}
      title={wired
        ? `open escalation ${escalation.id} in the Escalations tab`
        : 'navigation unavailable — this section was rendered without an onNavigate handler'}
      onClick={() => { if (onNavigate) onNavigate('esc', escalation.id); }}
    >
      <span className="mono">{escalation.id}</span>
      <span style={{ color: 'var(--fg-2)' }}>{escalation.summary}</span>
      <span style={{ color: 'var(--fg-3)' }}>
        L{escalation.level} · {escalation.severity}
      </span>
    </button>
  );
}

// ── One metric row ──
function MemoryEvalMetricRow({ metric, onNavigate }) {
  const m = metric;
  const trend = m.trend || { labels: [], values: [] };
  const Chart = chartForKind(m.kind);
  const gaps = trendGaps(trend.values);
  // A series with a hole cannot be drawn honestly by charts.jsx (see
  // trendGaps).  Equality, not an ordering comparison: nothing here re-derives
  // anything from a threshold.
  const plottable = Chart && gaps === 0;
  const labels = trend.labels || [];
  const span = labels.length
    ? `${labels[0]} → ${labels[labels.length - 1]}`
    : 'no runs';
  const badge = verdictBadge(m);
  return (
    <tr>
      <td className="mono" style={{ color: 'var(--fg-1)' }}>{m.metric_id}</td>
      <td className="mono" style={{ fontSize: 11, color: 'var(--fg-2)' }}>
        {m.kind}
        <div style={{ fontSize: 10, color: 'var(--fg-3)' }}>{m.rule_kind}</div>
      </td>
      <td>
        {/* verdict_detail carries the evaluator's own words, untranslated. */}
        <span className={badge.cls} title={m.verdict_detail || undefined}>
          {badge.label}
        </span>
        {m.limit_ref && (
          <div className="mono" style={{ fontSize: 10, color: 'var(--fg-3)' }}>
            {m.limit_ref}
          </div>
        )}
        {/* Under storm the per-metric escalations were deliberately collapsed
            into the one aggregate, so the absence of a link here is explained
            rather than left looking like a missing link. */}
        {m.parity === 'storm_collapsed'
          ? (
            <div className="mono" style={{ fontSize: 10, color: 'var(--fg-3)' }}>
              link suppressed — storm aggregate
            </div>
          )
          : m.escalation && (
            <div style={{ marginTop: 4 }}>
              <EscalationLink escalation={m.escalation} onNavigate={onNavigate} />
            </div>
          )}
        {/* Fingerprints are rendered whole — never parsed. They are the
            producer's private construction
            (memory_evals._escalation_projection()). */}
        {m.fingerprint && (
          <div className="mono" style={{ fontSize: 10, color: 'var(--fg-3)' }}>
            fp {m.fingerprint}
          </div>
        )}
      </td>
      {/* `value` is what the evaluator judged; `current_value` is what the
          metrics artifact says.  Two separate labelled fields — never
          conflated into one "the number". */}
      <td className="num">{dash(m.value)}</td>
      <td className="num">{dash(m.current_value)}</td>
      <td className="num">{dash(m.n)}</td>
      <td className="num">{dash(m.denominator)}</td>
      <td className="mono" style={{ fontSize: 11, color: 'var(--fg-2)' }}>{dash(m.direction)}</td>
      <td style={{ width: 160 }}>
        {/* Two DISTINCTLY worded suppression states, both reusing the
            "no chart, value only" shape chartForKind already establishes:
            an unknown kind is a rendering gap the payload files an
            `unknown_kind` issue for, whereas a hole is normal, fully-explained
            missing data.  Either way the row still shows value, current_value,
            n, denominator, direction and the verdict badge — the operator
            loses a 160px sparkline, never the signal. */}
        {plottable
          ? (
            <div style={{ height: 26 }} title={span}>
              {/* values passed through verbatim — never filtered or compacted */}
              <Chart values={trend.values} color={MEC.accent} />
            </div>
          )
          : !Chart
            ? (
              <span className="mono" style={{ fontSize: 10, color: 'var(--fg-3)' }}>
                no chart for kind {String(m.kind)}
              </span>
            )
            : (
              <span className="mono" style={{ fontSize: 10, color: 'var(--fg-3)' }}>
                no chart — {gaps} of {labels.length} runs produced no sample
              </span>
            )}
        <div className="mono" style={{ fontSize: 10, color: 'var(--fg-3)' }}>
          {trend.labels ? trend.labels.length : 0} pts
          {gaps ? ` · ${gaps} gap(s) — no chart drawn` : ''}
        </div>
      </td>
    </tr>
  );
}

// ── Limits provenance ──
//
// Collapsed by default (a <details>, open state persisted under
// 'df.memevals.prov' in the useOpenSet/usePersistedState idiom of
// tab_escalations.jsx:286) so provenance does not dominate the card.
//
// Everything here is DISPLAYED verbatim.  Nothing is compared, rounded into a
// verdict, or re-derived — see the verdictBadge comment (PRD section 8,
// G6/INV-5).  The label/value pairs are built as data rather than as JSX text
// so the field names live in quoted strings, which also keeps the
// no-comparison guard's member-access regex unambiguous.
const ME_PROV_OPEN_KEY = 'df.memevals.prov';

function readProvOpen() {
  try {
    return localStorage.getItem(ME_PROV_OPEN_KEY) === '1';
  } catch (e) {
    return false;
  }
}

function writeProvOpen(open) {
  try {
    localStorage.setItem(ME_PROV_OPEN_KEY, open ? '1' : '0');
  } catch (e) { /* private mode — the toggle simply does not persist */ }
}

function LimitsProvenance({ ev }) {
  const lim = ev.limits;
  if (!lim) {
    return (
      <div className="mono" style={{ fontSize: 10, color: 'var(--fg-3)' }}>
        no limits artifact for this eval — see the artifact issues notice above
      </div>
    );
  }
  const stamps = lim.baseline_run_stamps || [];
  const stampText = stamps.length
    ? `${stamps.slice(0, 3).join(', ')}${stamps.length > 3 ? ` … (${stamps.length} total)` : ''}`
    : '—';
  const rows = [
    ['alpha', lim.alpha],
    ['false_alarm_budget', lim.false_alarm_budget],
    ['runs_per_quarter', lim.runs_per_quarter],
    ['min_samples', lim.min_samples],
    ['baseline_window', lim.baseline_window],
    ['baseline_run_stamps', stampText],
    ['grandfather_set_hash', lim.grandfather_set_hash],
    ['run_stamp', lim.run_stamp],
    ['generator', lim.generator],
  ];
  return (
    <details open={readProvOpen()} onToggle={e => writeProvOpen(e.target.open)}>
      <summary className="mono" style={{ fontSize: 10, color: 'var(--fg-3)', cursor: 'pointer' }}>
        limits provenance
      </summary>
      {/* The provenance may have been stamped at an earlier run than the one
          on screen.  Saying so is not optional: listed flatly, an older alpha
          reads as governing the newer displayed run
          (memory_evals._read_limits() stamps `stale_for_latest_run`). */}
      {lim.stale_for_latest_run && (
        <div className="badge warn" style={{ margin: '4px 0' }}>
          provenance stamped at {lim.run_stamp} — does not govern{' '}
          {ev.latest_run_stamp}
        </div>
      )}
      <div
        className="mono"
        style={{
          display: 'grid',
          gridTemplateColumns: 'auto 1fr',
          gap: '2px 10px',
          fontSize: 10,
          color: 'var(--fg-3)',
          marginTop: 4,
        }}
      >
        {rows.map(r => [
          <span key={`${r[0]}-k`}>{r[0]}</span>,
          <span key={`${r[0]}-v`} style={{ color: 'var(--fg-2)' }}>{String(r[1])}</span>,
        ])}
      </div>
    </details>
  );
}

// ── Storm aggregate banner ──
//
// Reads the TOP-LEVEL storm_escape block — declared on every return path by
// memory_evals._empty_payload() and filled by memory_evals._build_payload() —
// never the copy repeated on an eval row: the top-level block is the single
// banner source, so the UI never has to elect a row to read it from and the
// banner survives a root with zero eval dirs.
function StormBanner({ storm, onNavigate }) {
  if (!storm) return null;
  return (
    <div
      className="badge bad"
      data-testid="memory-eval-storm-banner"
      style={{ width: '100%', justifyContent: 'flex-start', padding: '6px 10px', gap: 8, flexWrap: 'wrap' }}
    >
      <span>
        storm escape — {storm.alarm_count} alarms collapsed into one aggregate
        escalation
      </span>
      {storm.aggregate_fingerprint && (
        <span className="mono" style={{ color: 'var(--fg-3)' }}>
          fp {storm.aggregate_fingerprint}
        </span>
      )}
      {storm.escalation
        ? <EscalationLink escalation={storm.escalation} onNavigate={onNavigate} />
        : (
          <span className="mono" style={{ color: 'var(--fg-3)' }}>
            no open escalation carries this aggregate_fingerprint
          </span>
        )}
    </div>
  );
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

function UnmatchedEscalations({ rows, onNavigate }) {
  if (!rows || rows.length === 0) return null;
  return (
    <div className="panel">
      <div className="panel-head">
        <span className="title">Escalations with no matching metric row</span>
        <span className="meta">{rows.length}</span>
      </div>
      <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {rows.map(row => (
          <div key={row.id} style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <EscalationLink escalation={row} onNavigate={onNavigate} />
            <span className="mono" style={{ fontSize: 10, color: 'var(--fg-3)' }}>
              {row.reason} — {unmatchedReasonText(row.reason)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── One eval card ──
function MemoryEvalCard({ ev, onNavigate }) {
  const metrics = ev.metrics || [];
  const corpus = ev.corpus;
  return (
    <div className="panel">
      <div className="panel-head">
        <span className="title mono">{ev.eval_id}</span>
        {/* Stale is a HINT, never an alarm: the eval runner self-escalates on
            a missed run (PRD DD6/INV-5), and the threshold that decides
            `stale` lives server-side and never reaches this payload. */}
        {ev.stale && (
          <span
            className="badge muted"
            style={{ marginLeft: 8 }}
            title="displayed here for context; the eval runner is what reports a missed run"
          >
            stale — no run in a while
          </span>
        )}
        <span className="meta">
          {dash(ev.latest_run_stamp)} · {ageText(ev.latest_run_age_seconds)}
        </span>
      </div>
      <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {/* Truncation disclosure — names BOTH counts, so how much was dropped
            is visible rather than merely that something was. */}
        {ev.truncated && (
          <div className="mono" style={{ fontSize: 10, color: 'var(--fg-3)' }}>
            showing {ev.run_count} of {ev.runs_on_disk} runs on disk (truncated)
          </div>
        )}
        {corpus && (
          <div className="mono" style={{ fontSize: 10, color: 'var(--fg-3)', display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            {Object.keys(corpus).map(k => (
              <span key={k}>corpus.{k} {String(corpus[k])}</span>
            ))}
          </div>
        )}
        <table className="tbl">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Kind</th>
              <th>Verdict</th>
              <th className="num">Judged</th>
              <th className="num">Artifact</th>
              <th className="num">n</th>
              <th className="num">Denom</th>
              <th>Direction</th>
              <th>Trend</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map(m => (
              <MemoryEvalMetricRow key={m.metric_id} metric={m} onNavigate={onNavigate} />
            ))}
          </tbody>
        </table>
        <LimitsProvenance ev={ev} />
      </div>
    </div>
  );
}

// ── Artifact-issues notice ──
//
// Expanded by default, on purpose.  Collapsing a degraded-state notice
// reproduces the silent degradation it exists to prevent (INV-2/INV-4, the
// 2658 parse_failures precedent).  Each issue names its kind, eval_id, path
// and detail — a bare count tells the operator something is wrong but not what.
function IssuesNotice({ issues, issueCount }) {
  if (!(issueCount > 0)) return null;
  const rows = issues || [];
  return (
    <div
      data-testid="memory-eval-issues"
      style={{
        padding: '8px 12px',
        border: '1px solid var(--line)',
        borderRadius: 4,
        background: 'var(--bg-2)',
        color: 'var(--fg-2)',
        fontFamily: 'var(--mono)',
        fontSize: 11,
      }}
    >
      <div style={{ color: 'var(--warn)', marginBottom: 4 }}>
        {issueCount} artifact issue(s)
      </div>
      {rows.map((iss, i) => (
        <div key={`${iss.kind}-${iss.eval_id}-${i}`} style={{ color: 'var(--fg-3)' }}>
          {iss.kind} · {dash(iss.eval_id)} · {dash(iss.path)} — {dash(iss.detail)}
        </div>
      ))}
    </div>
  );
}

// ── The section ──
function MemoryEvalsSection({ onNavigate }) {
  const payload = MEDF.MEMORY_EVALS;
  const evals = payload.evals || [];
  return (
    <div className="grid cols-12" style={{ gap: 12 }}>
      <div className="col-span-12" style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
        <span className="lbl" style={{ color: 'var(--fg-3)', fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
          memory evals
        </span>
        <span className="mono" style={{ fontSize: 10, color: 'var(--fg-3)' }}>
          generated {dash(payload.generated_at)}
        </span>
      </div>
      {payload.issue_count > 0 && (
        <div className="col-span-12">
          <IssuesNotice issues={payload.issues} issueCount={payload.issue_count} />
        </div>
      )}
      {/* Two DISTINCT empty states. root_present false means the artifact tree
          does not exist yet; root_present true with zero evals is an
          empty-but-healthy system (memory_evals._build_payload()). Sharing one
          message would report a working system as a broken one. */}
      {!payload.root_present && (
        <div className="col-span-12 empty" data-testid="memory-eval-empty">
          no eval artifacts yet
        </div>
      )}
      {payload.root_present && evals.length === 0 && (
        <div className="col-span-12 empty">
          eval root present, no eval directories yet
        </div>
      )}
      {payload.storm_escape && (
        <div className="col-span-12">
          <StormBanner storm={payload.storm_escape} onNavigate={onNavigate} />
        </div>
      )}
      {evals.map(ev => (
        <div className="col-span-12" key={ev.eval_id}>
          <MemoryEvalCard ev={ev} onNavigate={onNavigate} />
        </div>
      ))}
      <div className="col-span-12">
        <UnmatchedEscalations rows={payload.unmatched_escalations} onNavigate={onNavigate} />
      </div>
    </div>
  );
}

window.DF_MEMORY_EVALS = { MemoryEvalsSection, chartForKind, verdictBadge };
