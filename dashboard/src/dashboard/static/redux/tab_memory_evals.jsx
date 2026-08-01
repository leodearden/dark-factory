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

// ── One metric row ──
function MemoryEvalMetricRow({ metric }) {
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
      <td className="mono" style={{ fontSize: 11, color: 'var(--fg-2)' }}>{m.kind}</td>
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

// ── One eval card ──
function MemoryEvalCard({ ev }) {
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
              <MemoryEvalMetricRow key={m.metric_id} metric={m} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── The section ──
function MemoryEvalsSection() {
  const payload = MEDF.MEMORY_EVALS;
  const evals = payload.evals || [];
  return (
    <div className="grid cols-12" style={{ gap: 12 }}>
      <div className="col-span-12" style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
        <span className="lbl" style={{ color: 'var(--fg-3)', fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
          memory evals
        </span>
      </div>
      {evals.map(ev => (
        <div className="col-span-12" key={ev.eval_id}>
          <MemoryEvalCard ev={ev} />
        </div>
      ))}
    </div>
  );
}

window.DF_MEMORY_EVALS = { MemoryEvalsSection, chartForKind, verdictBadge };
