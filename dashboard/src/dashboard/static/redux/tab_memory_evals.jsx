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
// means that run produced no sample; the array is handed to the chart
// UNMODIFIED (dropping a hole would shift this metric's points against every
// other metric's, since all series share the run_stamps x-axis), but the
// sparkline plots a hole at the baseline, so the gap count is disclosed in
// text beside it rather than left to read as a measured zero.
function trendGaps(values) {
  if (!values) return 0;
  let gaps = 0;
  for (let i = 0; i < values.length; i++) {
    if (values[i] === null || values[i] === undefined) gaps += 1;
  }
  return gaps;
}

// ── Verdict badge ──
//
// `verdict` and `parity` are the ONLY badge inputs.  Re-deriving alarm state
// from value-vs-limit in the browser is forbidden by PRD section 8 (G6/INV-5):
// memory_evals.py:660-661 says `parity` exists precisely so the UI does not
// re-derive badge state "out of three separate fields, which is where the two
// sides would drift apart".  This function therefore performs string equality
// only — no arithmetic, no ordering comparison, no limits.
//
// The parity dimension REFINES the verdict; it never replaces it.  Where the
// two agree there is nothing extra to say, so `alarmed_open` and `clear` fall
// through to the plain verdict badge.
function verdictBadge(metric) {
  const verdict = metric.verdict;
  const parity = metric.parity;

  // Parity states that carry information the verdict alone does not.
  if (parity === 'recovered_open') {
    return { cls: 'badge warn', label: 'recovered · escalation open' };
  }
  if (parity === 'alarmed_unlinked') {
    return { cls: 'badge bad', label: 'alarm · no escalation' };
  }
  if (parity === 'storm_collapsed') {
    return { cls: 'badge bad', label: 'alarm · storm-collapsed' };
  }
  // parity 'alarmed_open' / 'clear' agree with the verdict — plain badge.

  if (verdict === 'alarm') return { cls: 'badge bad', label: 'alarm' };
  if (verdict === 'no_alarm') return { cls: 'badge ok', label: 'no_alarm' };
  if (verdict === 'grandfathered') {
    return { cls: 'badge info', label: 'grandfathered' };
  }
  if (verdict === 'insufficient_data') {
    return { cls: 'badge muted', label: 'insufficient_data' };
  }
  // Absent is absent.  A null/unrecognised verdict is NEVER defaulted to
  // no_alarm — that would report "we did not measure" as "we measured and it
  // is fine" (mirrors memory_evals.py:847-849).
  return { cls: 'badge muted', label: 'no verdict' };
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
            producer's private construction (memory_evals.py:576-579). */}
        {m.fingerprint && (
          <div className="mono" style={{ fontSize: 10, color: 'var(--fg-3)' }}>
            fp {m.fingerprint}
          </div>
        )}
      </td>
      {/* `value` is what the evaluator judged; `current_value` is what the
          metrics artifact says.  Two separate labelled fields — never
          conflated into one "the number". */}
      <td className="num">{m.value}</td>
      <td className="num">{m.current_value}</td>
      <td className="num">{m.n}</td>
      <td className="num">{m.denominator}</td>
      <td className="mono" style={{ fontSize: 11, color: 'var(--fg-2)' }}>{m.direction}</td>
      <td style={{ width: 160 }}>
        {Chart
          ? (
            <div style={{ height: 26 }} title={span}>
              {/* values passed through verbatim — never filtered or compacted */}
              <Chart values={trend.values} color={MEC.accent} />
            </div>
          )
          : (
            <span className="mono" style={{ fontSize: 10, color: 'var(--fg-3)' }}>
              no chart for kind {String(m.kind)}
            </span>
          )}
        <div className="mono" style={{ fontSize: 10, color: 'var(--fg-3)' }}>
          {trend.labels ? trend.labels.length : 0} pts
          {gaps ? ` · ${gaps} gap(s) plotted at baseline` : ''}
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
          (memory_evals.py:237-241). */}
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
// Reads the TOP-LEVEL storm_escape block (memory_evals.py:958-964), never the
// copy repeated on an eval row: the top-level block is the single banner
// source, so the UI never has to elect a row to read it from and the banner
// survives a root with zero eval dirs.
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
// that catches a real parity orphan (memory_evals.py:530-534).
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
        <span className="meta">{ev.latest_run_stamp}</span>
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
      </div>
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
