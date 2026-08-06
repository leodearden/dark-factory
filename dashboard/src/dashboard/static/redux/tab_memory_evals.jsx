/* tab_memory_evals.jsx — memory-eval monitoring section, rendered INSIDE the
   Memory tab (PRD DD3: not a thirteenth top-level tab, no Rail entry).

   The JSX/React wiring here is verified via Python source-assertion tests in
   dashboard/tests/test_tab_memory_evals.py — no JSX runner exists (this file
   is transformed by CDN Babel at runtime and the repo has no node_modules).
   The pure, JSX-free logic that used to live here was moved to
   /static/redux/memory_evals_fmt.js (task 3481) so it could escape that
   limitation; WHY, in full, is in that file's header, which is the one place
   that account is kept.

   LOAD ORDER IS THE CONTRACT, in both directions.

   UPSTREAM — memory_evals_fmt.js must load BEFORE this file, because the
   destructure of window.DF_MEMORY_EVALS_FMT below runs at module top level.
   A later tag leaves that global undefined here, which throws, which leaves
   window.DF_MEMORY_EVALS undefined for tabs.jsx, which blanks every tab
   defined there — the downstream contract below, one link back.

   DOWNSTREAM — this file must load BEFORE tabs.jsx.
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

   Exports: window.DF_MEMORY_EVALS = { MemoryEvalsSection }
   (chartForKind and verdictBadge were exported here only so the Python suite
   could see them; they are owned by window.DF_MEMORY_EVALS_FMT now, and
   re-exporting them would create two paths to one function.  tabs.jsx, the
   only real consumer, destructures MemoryEvalsSection alone.)
*/
const { Sparkline: MESpark, StepSpark: MEStep, PALETTE: MEC } = window.DF_CHARTS;
const MEDF = window.DF_DATA;
// ME-prefixed like the two lines above.  The prefix is a readability
// convention here, NOT a collision workaround: this file is a
// `type="text/babel"` .jsx, and Babel-standalone downlevels .jsx top-level
// bindings so they never join the classic-script shared global lexical scope.
// That is an observed fact, not an assumption — tabs.jsx, tab_escalations.jsx
// and tab_escalation_analytics.jsx each already declare their own top-level
// `const DF` / `useOpenSet` / `usePersistedState` and all render fine.  See the
// SCOPE note in dashboard/tests/js/classic_script_scope.test.mjs before
// "fixing" this.
const { useState: MEuS } = React;

// The pure, JSX-free helpers live in memory_evals_fmt.js (task 3481); their
// behavioural suite is dashboard/tests/js/memory_evals_fmt.test.mjs.
// index.html loads that classic script before this file, exactly as this file
// loads before tabs.jsx.
//
// NOTE `chartForKind` returns a TAG ('step' | 'spark' | null), NOT a component
// — see its comment in memory_evals_fmt.js.  ME_CHART_BY_TAG below is the
// two-entry lookup that turns the tag back into a primitive.
const {
  chartForKind,
  trendGaps,
  dash,
  ageText,
  verdictBadge,
  unmatchedReasonText,
} = window.DF_MEMORY_EVALS_FMT;

// Tag -> chart primitive.  An unknown kind yields the `null` tag, which misses
// this table and leaves `Chart` null, preserving the `plottable` truthiness
// gate below: value only, NO chart.
const ME_CHART_BY_TAG = { step: MEStep, spark: MESpark };

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
  const Chart = ME_CHART_BY_TAG[chartForKind(m.kind)] || null;
  const gaps = trendGaps(trend.values);
  // An EMPTY series is not a drawable series: both Sparkline and StepSpark
  // return null for a zero-length array, so without this count a metric with
  // no runs renders a blank 26px box.  `trendGaps([])` is 0, so the gap check
  // alone cannot tell the two apart.
  //
  // `points` is the ONE series count in this row — the chart gate, the gap
  // disclosure and the footer all read it.  Measuring the state from
  // trend.values and the footer from trend.labels would let a payload where
  // they disagree render the contradictory pair "no runs yet" next to "N pts",
  // which is exactly the reads-as-a-bug outcome the no-runs state was added to
  // prevent, one line further down.
  const labels = trend.labels || [];
  const points = (trend.values || []).length;
  // labels and values are PARALLEL arrays — one entry per run, both built from
  // the same `runs` list server-side (memory_evals.py:955,993).  A payload
  // where they disagree cannot be reconciled here: neither length is more
  // authoritative, the chart would be drawn against a `span` title derived from
  // the other array, and the gap message would print "1 of 0 runs".  It gets
  // its own named state rather than a silently-picked winner.
  const seriesMismatch = labels.length !== points;
  // A series with a hole cannot be drawn honestly by charts.jsx (see
  // trendGaps).  Equality, not an ordering comparison: nothing here re-derives
  // anything from a threshold.
  const plottable = Chart && gaps === 0 && points > 0 && !seriesMismatch;
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
        {/* FOUR DISTINCTLY worded suppression states, all reusing the
            "no chart, value only" shape chartForKind already establishes.
            They assert different things and must not be collapsed:

              * unknown kind — a RENDERING gap; the payload passes the value
                through verbatim and files an `unknown_kind` issue for it.
              * length disagreement — a MALFORMED payload: labels and values are
                parallel arrays, so nothing else this cell could say about the
                series would be trustworthy. Named, never silently reconciled
                by picking one length over the other.
              * holed series — normal, fully-explained MISSING DATA: some runs
                produced no sample, and the count says how many.
              * no runs — the metric simply has NOT BEEN MEASURED yet. Folding
                this into the gap message would print "0 of 0 runs produced no
                sample", a nonsense sentence that reads as a bug.

            `!Chart` stays first: an unrenderable kind is the more actionable
            fact than an empty or malformed series.  In every case the row still
            shows value, current_value, n, denominator, direction and the
            verdict badge — the operator loses a 160px sparkline, never the
            signal. */}
        {plottable
          ? (
            <div style={{ height: 26 }} title={span} data-testid="memory-eval-trend-chart">
              {/* values passed through verbatim — never filtered or compacted */}
              <Chart values={trend.values} color={MEC.accent} />
            </div>
          )
          : !Chart
            ? (
              <span
                className="mono"
                style={{ fontSize: 10, color: 'var(--fg-3)' }}
                data-testid="memory-eval-trend-no-kind"
              >
                no chart for kind {String(m.kind)}
              </span>
            )
            : seriesMismatch
              ? (
                <span
                  className="mono"
                  style={{ fontSize: 10, color: 'var(--fg-3)' }}
                  data-testid="memory-eval-trend-mismatch"
                >
                  no chart — {labels.length} run labels but {points} samples
                </span>
              )
              : points === 0
                ? (
                  <span
                    className="mono"
                    style={{ fontSize: 10, color: 'var(--fg-3)' }}
                    data-testid="memory-eval-trend-no-runs"
                  >
                    no runs yet — nothing to chart
                  </span>
                )
                : (
                  <span
                    className="mono"
                    style={{ fontSize: 10, color: 'var(--fg-3)' }}
                    data-testid="memory-eval-trend-gaps"
                  >
                    no chart — {gaps} of {labels.length} runs produced no sample
                  </span>
                )}
        {/* Footer count reads `points`, the SAME local the states above are
            derived from — never the labels array's own length, which would
            contradict the no-runs state whenever the two arrays disagree. */}
        <div className="mono" style={{ fontSize: 10, color: 'var(--fg-3)' }}>
          {points} pts
          {gaps ? ` · ${gaps} gap(s) — no chart drawn` : ''}
        </div>
      </td>
    </tr>
  );
}

// ── Limits provenance ──
//
// Collapsed by default (a <details>, open state persisted in the
// useOpenSet/usePersistedState idiom of tab_escalations.jsx:286) so provenance
// does not dominate the card.
//
// The persisted key is PER EVAL — `df.memevals.prov.<eval_id>` — and the open
// state is held in component state seeded from storage exactly once at mount.
// Both halves matter, and both were once wrong:
//
//   * One shared key meant expanding ONE card's provenance wrote '1' for
//     every card, so the next poll-driven re-render expanded all of them.
//     A <details> open state is per-disclosure UI state; keying it on the
//     section makes one operator's click read as a rendering bug everywhere
//     else.
//   * Calling the reader inside the JSX attribute (`open={readProvOpen()}`)
//     re-ran a synchronous localStorage read on EVERY render — once per eval
//     card per 3s poll tick — and made the toggle unrecoverable, since each
//     poll overwrote whatever the operator had just opened.
//
// No migration off the old flat 'df.memevals.prov' key: a lost <details> open
// state is a cosmetic default, not data.
//
// Everything here is DISPLAYED verbatim.  Nothing is compared, rounded into a
// verdict, or re-derived — see the verdictBadge comment (PRD section 8,
// G6/INV-5).  The label/value pairs are built as data rather than as JSX text
// so the field names live in quoted strings, which also keeps the
// no-comparison guard's member-access regex unambiguous.
const ME_PROV_OPEN_PREFIX = 'df.memevals.prov';

function provOpenKey(evalId) {
  return `${ME_PROV_OPEN_PREFIX}.${evalId}`;
}

// The key is a PARAMETER, not a closed-over module constant: that is what
// makes one global key shared by every card structurally unrepresentable.
function readProvOpen(key) {
  try {
    return localStorage.getItem(key) === '1';
  } catch (e) {
    return false;
  }
}

function writeProvOpen(key, open) {
  try {
    localStorage.setItem(key, open ? '1' : '0');
  } catch (e) { /* private mode — the toggle simply does not persist */ }
}

function LimitsProvenance({ ev }) {
  // ABOVE the `if (!lim)` early return below — that guard is a conditional
  // return, so a hook placed after it would change the hook count on the first
  // eval with no limits artifact (Rules of Hooks) and blank the card.
  const provKey = provOpenKey(ev.eval_id);
  const [provOpen, setProvOpen] = MEuS(() => readProvOpen(provKey));
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
    <details
      open={provOpen}
      onToggle={e => { setProvOpen(e.target.open); writeProvOpen(provKey, e.target.open); }}
    >
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

window.DF_MEMORY_EVALS = { MemoryEvalsSection };
