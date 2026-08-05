// Behavioural tests for memory_evals_fmt.js — the plain-JS (no JSX/Babel)
// module holding the pure display/vocabulary helpers behind the memory-eval
// monitoring section rendered by tab_memory_evals.jsx. Run via `node --test`
// (see dashboard/tests/test_graph_layout_js.py for the pytest wrapper that
// surfaces this suite in CI via its `**/*.test.mjs` glob — no wrapper change
// needed for this new file).
//
// WHY THIS FILE EXISTS (task 3481) — tab_memory_evals.jsx is JSX transformed by
// CDN Babel at runtime and this repo has no node_modules, so node cannot parse
// it and React cannot be rendered in any harness here. Six JSX-FREE helpers
// nonetheless lived inside that .jsx — a verdict×parity badge matrix, an age
// formatter, a null-vs-zero placeholder, a trend-hole counter, a chart-kind
// vocabulary and an unmatched-reason branch — and the only reachable test for
// them was a Python suite that read the .jsx AS TEXT and asserted regexes over
// it. That idiom absorbed review cycles 3-6 of task 3216 and is structurally
// weak in three ways this suite fixes:
//
//   1. It never EXECUTES the code. `verdictBadge` composes a base label with a
//      parity suffix across the 7 verdict inputs that produce a distinct base
//      (the 4 in-vocabulary verdicts, absent-as-null, absent-as-undefined, and
//      a present-but-unreadable one) × 13 parity states = 91 combinations; a
//      regex can see that a `+ ' · ' +` exists, never that the right string
//      comes out of the right pair.
//   2. A regex that matches NOTHING passes silently. Two regexes in that file
//      were in fact found matching nothing at all (hence the `assert body` /
//      "would pass vacuously" guards littered through it).
//   3. Vocabulary assertions drifted from a hand-pinned COPY of the parity
//      table rather than the real one — tab_memory_evals.jsx records that "a
//      hand-picked three-member copy DID live in that test, and went blind to
//      six states". This suite drives its matrix off the module's EXPORTED
//      tables, so a state the producer adds is covered the moment it lands.
//
// What stays in Python is exactly what node cannot do: cross-language
// completeness against the PYTHON frozenset `memory_evals.PARITY_STATES`, the
// PRD-section-8 G6/INV-5 source guard, index.html load order, and JSX/React
// render wiring.
//
// memory_evals_fmt.js has no package.json in the repo, so it resolves as
// CommonJS (`module.exports = <object>`). Node's cjs-module-lexer cannot
// statically detect named exports assigned from a variable, so
// `import { dash } from '...'` would come back undefined. We therefore
// default-import the module and destructure instead (mirrors
// spark_path.test.mjs / runtime_format.test.mjs / prd_grouping.test.mjs).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import mef from '../../src/dashboard/static/redux/memory_evals_fmt.js';

const {
  dash,
  ageText,
  unmatchedReasonText,
  chartForKind,
  trendGaps,
  verdictBadge,
  PARITY_REFINEMENT,
  PARITY_PLAIN,
} = mef;

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/memory_evals_fmt.js';

// The module's complete public surface, split by kind because the parity
// TABLES are exported as data alongside the functions (so this suite can
// enumerate the real vocabulary instead of pinning a copy). Asserted
// EXHAUSTIVELY below as the union of the two lists: an accidental export would
// otherwise quietly widen the contract tab_memory_evals.jsx depends on.
const EXPECTED_FUNCTION_NAMES = [
  'ageText',
  'chartForKind',
  'dash',
  'trendGaps',
  'unmatchedReasonText',
  'verdictBadge',
];
// The parity vocabulary TABLES are exported as data so this suite can drive its
// verdict×parity matrix off the REAL vocabulary the module ships, rather than a
// copy pinned here. tab_memory_evals.jsx records why that matters: "a
// hand-picked three-member copy DID live in that test, and went blind to six
// states". A state the producer adds is covered the moment it lands in the table.
const EXPECTED_DATA_NAMES = ['PARITY_PLAIN', 'PARITY_REFINEMENT'];

function expectedSurface() {
  return EXPECTED_FUNCTION_NAMES.concat(EXPECTED_DATA_NAMES).slice().sort();
}

// The em-dash placeholder, spelled by codepoint so a copy-paste of a hyphen or
// an en-dash into either file fails here rather than rendering as a near-miss.
const EM_DASH = '—';

// ---------------------------------------------------------------------------
// Module surface and dual export
// ---------------------------------------------------------------------------

test('default-imported module exposes exactly the memory-eval format helpers', () => {
  assert.deepEqual(
    Object.keys(mef).sort(),
    expectedSurface(),
    'the public surface must be exactly these names — an accidental export ' +
      'widens the contract silently',
  );
  for (const name of EXPECTED_FUNCTION_NAMES) {
    assert.equal(typeof mef[name], 'function', `mef.${name} should be a function`);
  }
  for (const name of EXPECTED_DATA_NAMES) {
    assert.equal(typeof mef[name], 'object', `mef.${name} should be a data table`);
    assert.ok(mef[name] !== null, `mef.${name} should not be null`);
  }
});

test('module also assigns window.DF_MEMORY_EVALS_FMT (browser dual-export)', () => {
  // Shim a bare browser-like global before requiring the module fresh via
  // CommonJS require, so the module body's `if (typeof window !== 'undefined')`
  // branch executes against our shim.
  globalThis.window = {};
  try {
    const require = createRequire(import.meta.url);
    // Node's ESM loader resolves a CommonJS module (no package.json/type in
    // this repo) by delegating to the CJS loader and populating the shared
    // require.cache — so by the time this test runs, the top-level
    // `import mef from ...` above has ALREADY cached this exact file. A plain
    // require() here would return that cached module.exports without
    // re-running the module body, meaning the dual-export line would never see
    // our globalThis.window shim. Busting the cache entry forces a fresh
    // execution against the now-shimmed window.
    const resolved = require.resolve(MODULE_SPECIFIER);
    delete require.cache[resolved];
    const required = require(MODULE_SPECIFIER);

    assert.ok(globalThis.window.DF_MEMORY_EVALS_FMT, 'window.DF_MEMORY_EVALS_FMT was not set');

    // The fresh require() and the top-level import() produce two distinct API
    // object instances (separate module executions), so compare structurally
    // — same set of exported names — rather than asserting reference/deep
    // equality against the ESM-imported `mef`.
    assert.deepEqual(Object.keys(globalThis.window.DF_MEMORY_EVALS_FMT).sort(), expectedSurface());
    assert.deepEqual(Object.keys(required).sort(), expectedSurface());

    // tab_memory_evals.jsx destructures this global at module top level, so
    // every name it reaches for must be present on the BROWSER export path
    // specifically — not merely on the CommonJS one.
    for (const name of EXPECTED_FUNCTION_NAMES) {
      assert.equal(
        typeof globalThis.window.DF_MEMORY_EVALS_FMT[name],
        'function',
        `window.DF_MEMORY_EVALS_FMT.${name} should be a function`,
      );
      assert.equal(typeof required[name], 'function', `required.${name} should be a function`);
    }
  } finally {
    delete globalThis.window;
  }
});

// ---------------------------------------------------------------------------
// dash — the null-vs-zero placeholder.
//
// The load-bearing property is NOT "missing renders as a dash"; it is that a
// measured 0 does NOT. `|| 0` (or a `!v` guard) would make an absent scalar
// indistinguishable from a metric that genuinely measured zero, which for a
// proportion metric reads as total failure. That distinction is asserted
// STRICTLY here: dash(0) must be the NUMBER 0, not '0' and not the em-dash.
// ---------------------------------------------------------------------------

test('dash: null renders as the em-dash', () => {
  assert.equal(dash(null), EM_DASH);
});

test('dash: undefined renders as the em-dash', () => {
  assert.equal(dash(undefined), EM_DASH);
});

test('dash: an honest zero is preserved as the number 0, never dashed', () => {
  const out = dash(0);
  assert.equal(out, 0);
  assert.equal(typeof out, 'number', 'a measured zero must stay a number, not become a string');
  assert.notEqual(out, EM_DASH);
});

test('dash: other falsy-but-measured values pass through unchanged', () => {
  // `false` and '' are as distinguishable from "absent" as 0 is; a truthiness
  // guard would collapse all three into the placeholder.
  assert.equal(dash(false), false);
  assert.equal(dash(''), '');
});

test('dash: a negative value passes through unchanged', () => {
  assert.equal(dash(-1), -1);
});

test('dash: a positive value passes through unchanged', () => {
  assert.equal(dash(0.95), 0.95);
});

// ---------------------------------------------------------------------------
// ageText — compact age from `latest_run_age_seconds`. Display only: the
// staleness THRESHOLD lives server-side and is deliberately absent from the
// payload, so nothing here can re-derive `stale`. The `h < 1` / `h < 48`
// comparisons below are DISPLAY unit boundaries, not judgments about a metric
// value against a limit (see the G6/INV-5 guard in
// dashboard/tests/test_tab_memory_evals.py, which matches on threshold-FIELD
// member access rather than banning `<` outright, precisely so these are legal).
//
// The boundaries are read off the source and asserted at the exact hinge
// values, where an off-by-one between `<` and `<=` is visible.
// ---------------------------------------------------------------------------

test('ageText: an absent age renders as the em-dash, never as "0m ago"', () => {
  assert.equal(ageText(null), EM_DASH);
  assert.equal(ageText(undefined), EM_DASH);
});

test('ageText: a zero age is a measured zero, rendered in minutes', () => {
  assert.equal(ageText(0), '0m ago');
});

test('ageText: sub-hour ages round to the nearest minute', () => {
  assert.equal(ageText(59), '1m ago'); // Math.round(59/60) === 1
  assert.equal(ageText(60), '1m ago');
  assert.equal(ageText(90), '2m ago'); // Math.round(1.5) === 2
  assert.equal(ageText(3599), '60m ago');
});

test('ageText: exactly one hour crosses into the hours unit (h < 1 is false at h === 1)', () => {
  assert.equal(ageText(3600), '1h ago');
});

test('ageText: the last hour before the day boundary still reads in hours', () => {
  assert.equal(ageText(47 * 3600), '47h ago');
});

test('ageText: exactly 48h crosses into the days unit (h < 48 is false at h === 48)', () => {
  assert.equal(ageText(172800), '2d ago');
});

test('ageText: beyond 48h reads in days, rounded', () => {
  assert.equal(ageText(49 * 3600), '2d ago'); // Math.round(49/24) === 2
  assert.equal(ageText(72 * 3600), '3d ago');
});

// ---------------------------------------------------------------------------
// unmatchedReasonText — the three reasons an escalation can land in the
// "no matching metric row" list, each with DISTINCT wording. Collapsing them
// into one undifferentiated "unexplained" list would fire on escalations that
// are in fact fully explained and train operators to ignore the one signal
// that catches a real parity orphan (memory_evals._unmatched_projection()).
//
// This is the executable replacement for the regex at
// test_tab_memory_evals.py:1724-1750, which recognised only single-quoted
// ternaries and therefore silently matched NOTHING for a body that uses `if`
// returns and a double-quoted string — the vacuous-pass failure mode this
// whole migration exists to end.
// ---------------------------------------------------------------------------

test('unmatchedReasonText: each known reason gets its own non-blank wording', () => {
  const known = ['no_matching_verdict', 'storm_suppressed', 'no_fingerprint'];
  const texts = known.map(unmatchedReasonText);

  for (let i = 0; i < known.length; i++) {
    assert.equal(typeof texts[i], 'string', `${known[i]} should render a string`);
    assert.ok(texts[i].trim().length > 0, `${known[i]} rendered blank`);
    // A known reason must never fall through to the unrecognised branch.
    assert.ok(
      !texts[i].startsWith('unrecognised reason'),
      `${known[i]} fell through to the unrecognised branch`,
    );
  }

  // Pairwise DISTINCT: three reasons rendering the same words is the collapse
  // the branch exists to prevent, and would otherwise pass every per-reason
  // assertion above.
  assert.equal(new Set(texts).size, known.length, 'reason wordings must be pairwise distinct');
});

test('unmatchedReasonText: no_matching_verdict names the missing metric row', () => {
  assert.equal(unmatchedReasonText('no_matching_verdict'), 'no metric row explains this');
});

test('unmatchedReasonText: storm_suppressed says the links are collapsed, not missing', () => {
  assert.equal(
    unmatchedReasonText('storm_suppressed'),
    "explained, but this run's links are collapsed into the aggregate",
  );
});

test('unmatchedReasonText: no_fingerprint names the absent dedupe_fingerprint', () => {
  assert.equal(unmatchedReasonText('no_fingerprint'), 'producer emitted no dedupe_fingerprint');
});

test('unmatchedReasonText: an unrecognised reason is marked and echoed', () => {
  assert.equal(unmatchedReasonText('brand_new_reason'), 'unrecognised reason: brand_new_reason');
});

test('unmatchedReasonText: absent reasons coerce visibly rather than crashing or blanking', () => {
  // String() coercion is explicit in the source for exactly this: a bare
  // template interpolation of null would still read 'null', but an absent
  // reason must never produce a bare 'unrecognised reason: ' with nothing
  // after it, and must never throw on a payload the producer shortened.
  assert.equal(unmatchedReasonText(null), 'unrecognised reason: null');
  assert.equal(unmatchedReasonText(undefined), 'unrecognised reason: undefined');
  assert.equal(unmatchedReasonText(''), 'unrecognised reason: ');
});

// ---------------------------------------------------------------------------
// chartForKind — the metric-kind → chart-primitive vocabulary.
//
// RETURNS A TAG, NOT A COMPONENT. The function used to return MEStep/MESpark
// directly, React component references destructured from window.DF_CHARTS,
// which is exactly what pinned it inside the .jsx and out of reach of any
// runner. It now returns a closed-vocabulary tag string and
// tab_memory_evals.jsx maps tag→component at its single call site, so the
// vocabulary and its fallback — the actual decision content — are executable
// here (task 3481, DD-1).
//
// The payload's kind vocabulary is exactly {tripwire, proportion, count,
// scalar}. A kind outside that set is a RENDERING gap, not a data error, so
// the fallback is null — value only, NO chart.
// ---------------------------------------------------------------------------

test('chartForKind: tripwire is step-shaped', () => {
  assert.equal(chartForKind('tripwire'), 'step');
});

// The three sparkline kinds are asserted SEPARATELY rather than as a group, so
// dropping one from the vocabulary fails a named test instead of thinning a
// loop nobody reads.
test('chartForKind: proportion is a sparkline', () => {
  assert.equal(chartForKind('proportion'), 'spark');
});

test('chartForKind: count is a sparkline', () => {
  assert.equal(chartForKind('count'), 'spark');
});

test('chartForKind: scalar is a sparkline', () => {
  assert.equal(chartForKind('scalar'), 'spark');
});

test('chartForKind: an unknown kind gets NO chart, never a guessed primitive', () => {
  // A kind outside the closed vocabulary means the builder passed a value
  // through verbatim and filed an `unknown_kind` issue for it. Guessing a
  // primitive would render an unvalidated shape as though it were understood.
  for (const kind of ['histogram', 'ratio', 'TRIPWIRE', '', null, undefined, 0]) {
    assert.equal(
      chartForKind(kind),
      null,
      `kind ${JSON.stringify(kind)} must resolve to no chart at all`,
    );
  }
});

test('chartForKind: the tag vocabulary is exactly {step, spark, null}', () => {
  // The .jsx maps these tags through a two-entry lookup; a tag outside this
  // set would silently resolve to undefined there and disable the chart with
  // no diagnosis.
  const tags = ['tripwire', 'proportion', 'count', 'scalar'].map(chartForKind);
  assert.deepEqual([...new Set(tags)].sort(), ['spark', 'step']);
});

// ---------------------------------------------------------------------------
// trendGaps — count the deliberate holes in a trend series. A `null` (or
// `undefined`) in `trend.values` means that run produced no sample.
//
// DETECT WITHOUT DROPPING is the load-bearing property. The array is handed on
// UNMODIFIED: all metrics share one `run_stamps` x-axis, so compacting one
// series would shift its points against every other's. The Python suite could
// only approach this negatively (grep for the absence of `.filter(Boolean)`);
// here it is asserted directly — the return is a NUMBER, and the input array is
// deep-equal to a pristine copy afterwards. That is what a `.filter(Boolean)`,
// an in-place `splice`, or a "helpfully" compacting rewrite would break.
//
// It matters because charts.jsx's primitives coerce a null to 0, and a dropped
// hole would be drawn as a real measured point at the chart floor.
// ---------------------------------------------------------------------------

test('trendGaps: a clean series has no gaps', () => {
  assert.equal(trendGaps([1, 2, 3, 0, -4]), 0);
});

test('trendGaps: both null and undefined count as holes', () => {
  assert.equal(trendGaps([1, null, 3, undefined, 5]), 2);
});

test('trendGaps: an empty series has no gaps', () => {
  // Note an empty series is NOT the same state as a holed one: `trendGaps([])`
  // is 0, so the caller gates the chart on the point count separately.
  assert.equal(trendGaps([]), 0);
});

test('trendGaps: an all-holes series reports every slot', () => {
  const values = [null, null, undefined, null];
  assert.equal(trendGaps(values), values.length);
});

test('trendGaps: a measured zero is not a hole', () => {
  // The same null-vs-zero invariant dash() states for scalars, applied to the
  // trend column: a run that measured 0 took a measurement.
  assert.equal(trendGaps([0, 0, 0]), 0);
});

test('trendGaps: an absent series is not an error', () => {
  // The `if (!values) return 0` guard — a payload with no trend at all must
  // render as "no runs", not throw through the row.
  assert.equal(trendGaps(undefined), 0);
  assert.equal(trendGaps(null), 0);
});

test('trendGaps: counts holes WITHOUT dropping them from the series', () => {
  const values = [1, null, 3, undefined, 5];
  const pristine = values.slice();

  const gaps = trendGaps(values);

  // (a) The result is a COUNT, not a compacted series — a function that
  // returned an array would have become a filter.
  assert.equal(typeof gaps, 'number', 'trendGaps must return a count, not a series');
  assert.equal(gaps, 2);

  // (b) The caller's array is untouched. Holes stay at their own indices
  // because every metric's series is indexed against one shared run_stamps
  // x-axis; compacting this one would shift its points against every other's.
  assert.deepEqual(values, pristine, 'the input series must not be mutated');
  assert.equal(values.length, pristine.length, 'the series length must not change');
  assert.equal(values[1], null, 'the null hole must stay at its own index');
  assert.equal(values[3], undefined, 'the undefined hole must stay at its own index');
});

// ---------------------------------------------------------------------------
// verdictBadge — the verdict×parity matrix.
//
// `verdict` and `parity` are the ONLY badge inputs. Re-deriving alarm state
// from value-vs-limit in the browser is forbidden by PRD section 8 (G6/INV-5);
// the source-level guard for that lives in Python
// (test_tab_memory_evals.py::test_no_client_side_alarm_derivation, which scans
// BOTH this module and tab_memory_evals.jsx). What this suite adds is the half
// a regex could never reach: which label and class each COMBINATION actually
// produces.
//
// The state axis is built from the module's EXPORTED tables, never a copy — see
// EXPECTED_DATA_NAMES above.
// ---------------------------------------------------------------------------

// The verdict axis, with the base label and class each verdict must yield on
// its own. Asserted directly in the base-label test below, then reused as the
// expected PREFIX for every parity combination — which is what makes
// "compose, never replace" checkable across the whole matrix at once.
const VERDICT_BASES = [
  { verdict: 'alarm', base: 'alarm', cls: 'badge bad' },
  { verdict: 'no_alarm', base: 'no_alarm', cls: 'badge ok' },
  { verdict: 'grandfathered', base: 'grandfathered', cls: 'badge info' },
  { verdict: 'insufficient_data', base: 'insufficient_data', cls: 'badge muted' },
  // Genuinely ABSENT — memory_evals._verdict_class() buckets this to `unjudged`.
  { verdict: null, base: 'no verdict', cls: 'badge muted' },
  { verdict: undefined, base: 'no verdict', cls: 'badge muted' },
  // PRESENT but outside the closed vocabulary — the `unknown_verdict` bucket.
  { verdict: 'wat', base: 'unreadable verdict', cls: 'badge bad' },
];

function allParityStates() {
  return Object.keys(PARITY_REFINEMENT).concat(PARITY_PLAIN);
}

test('verdictBadge: the exported parity vocabulary is the full 13-state split', () => {
  // 9 refined + 4 plain = 13 = |memory_evals.PARITY_STATES|. The cross-language
  // half of this — that those 13 are exactly the producer's frozenset — stays in
  // Python, which is the only place that frozenset can be imported.
  assert.equal(Object.keys(PARITY_REFINEMENT).length, 9, 'expected 9 refined states');
  assert.equal(PARITY_PLAIN.length, 4, 'expected 4 plain states');
  assert.equal(allParityStates().length, 13);

  // The two declarations must be DISJOINT: a state in both would be ambiguous
  // about whether it refines the badge or declines to.
  const overlap = PARITY_PLAIN.filter(s =>
    Object.prototype.hasOwnProperty.call(PARITY_REFINEMENT, s),
  );
  assert.deepEqual(overlap, [], 'a state may be refined or plain, never both');
  assert.equal(new Set(allParityStates()).size, 13, 'parity states must be unique');
});

test('verdictBadge: each verdict alone yields its own base label and class', () => {
  for (const { verdict, base, cls } of VERDICT_BASES) {
    const out = verdictBadge({ verdict, parity: null });
    assert.equal(out.label, base, `verdict ${JSON.stringify(verdict)} label`);
    assert.equal(out.cls, cls, `verdict ${JSON.stringify(verdict)} class`);
  }
});

test('verdictBadge: absent and unreadable are DIFFERENT, and unreadable is never muted', () => {
  const absent = verdictBadge({ verdict: null, parity: null });
  const unreadable = verdictBadge({ verdict: 'not_a_verdict', parity: null });

  // 'no verdict' asserts the verdict is MISSING. Saying it for a value that is
  // present but out-of-vocabulary would be a false statement about the payload.
  assert.notEqual(absent.label, unreadable.label);
  assert.equal(absent.label, 'no verdict');
  assert.equal(unreadable.label, 'unreadable verdict');

  // The producer files a named `unknown_verdict` issue for exactly this value;
  // the last render step must fail toward "something is wrong here", not toward
  // the muted/healthy end.
  assert.equal(unreadable.cls, 'badge bad');
  assert.notEqual(unreadable.cls, 'badge muted');
  assert.notEqual(unreadable.cls, 'badge ok');
});

test('verdictBadge: no parity branch can discard the verdict (compose, never replace)', () => {
  // The structural statement of CONSTRAINT (1), across the WHOLE matrix:
  // 7 verdicts × 13 parity states. The parity dimension REFINES the verdict; a
  // branch returning a FIXED label would collapse distinctions the producer
  // drew deliberately.
  let checked = 0;
  for (const { verdict, base } of VERDICT_BASES) {
    for (const parity of allParityStates()) {
      const out = verdictBadge({ verdict, parity });
      assert.ok(
        out.label === base || out.label.startsWith(base + ' · '),
        `verdict ${JSON.stringify(verdict)} × parity '${parity}' produced ` +
          `'${out.label}', which does not carry the verdict base '${base}'`,
      );
      assert.ok(out.cls, `verdict ${JSON.stringify(verdict)} × parity '${parity}' has no class`);
      checked += 1;
    }
  }
  assert.equal(checked, VERDICT_BASES.length * 13, 'the matrix must be fully enumerated');
});

test('verdictBadge: an escalation-open parity on a never-measured metric is NOT a recovery', () => {
  // Regression case for commit f79daa4c06. memory_evals._parity() suffixes
  // `_open` onto the linked variant of EVERY non-alarm verdict class, so an
  // `insufficient_data` metric with a linked escalation reaches
  // `insufficient_data_open` — and previously reached `recovered_open`. A
  // parity-FIRST branch would render "recovered", telling the operator a metric
  // nothing ever measured had recovered: the "we did not measure" -> "we
  // measured and it is fine" substitution this ordering exists to prevent.
  const out = verdictBadge({ verdict: 'insufficient_data', parity: 'recovered_open' });
  assert.equal(out.label, 'insufficient_data · escalation open');
  assert.ok(!out.label.includes('recovered'), 'must not read as a recovery');
  assert.equal(out.cls, 'badge warn');

  // The same metric via its own state reads identically — the two paths agree
  // because both suffix the verdict-derived base.
  const own = verdictBadge({ verdict: 'insufficient_data', parity: 'insufficient_data_open' });
  assert.equal(own.label, 'insufficient_data · escalation open');
});

test('verdictBadge: every _open state renders the same escalation-open affordance', () => {
  // `_parity()` suffixes `_open` onto the linked variant of every non-alarm
  // verdict class, so these all assert ONE fact — an escalation is filed and
  // still live — and the operator should read one marker rather than learn six.
  const openStates = Object.keys(PARITY_REFINEMENT).filter(s => s.endsWith('_open'));
  assert.ok(openStates.length >= 6, `expected the _open family, got ${openStates.length}`);

  for (const parity of openStates) {
    assert.equal(
      PARITY_REFINEMENT[parity].suffix,
      'escalation open',
      `${parity} must use the shared escalation-open wording`,
    );
    const out = verdictBadge({ verdict: 'no_alarm', parity });
    assert.ok(out.label.endsWith(' · escalation open'), `${parity} label: ${out.label}`);
    // An open escalation is never a healthy state, whatever the verdict says.
    assert.notEqual(out.cls, 'badge ok', `${parity} must not render as healthy`);
  }
});

test('verdictBadge: severity follows the verdict, not the linkage', () => {
  // An open escalation on a metric nothing judged alarming is a warn-level
  // parity fact, not a healthy one and not an alarm.
  const warn = ['recovered_open', 'insufficient_data_open', 'grandfathered_open', 'unjudged_open'];
  for (const parity of warn) {
    assert.equal(PARITY_REFINEMENT[parity].cls, 'badge warn', `${parity} should be warn`);
    assert.equal(verdictBadge({ verdict: 'no_alarm', parity }).cls, 'badge warn');
  }

  // `alarmed_open` keeps `badge bad` and pairs symmetrically with
  // `alarmed_unlinked` — "alarm · escalation open" vs "alarm · no escalation" —
  // where an unsuffixed alarm badge was ambiguous between "escalation linked"
  // and "state nobody handled".
  const bad = ['alarmed_open', 'alarmed_unlinked', 'storm_collapsed'];
  for (const parity of bad) {
    assert.equal(PARITY_REFINEMENT[parity].cls, 'badge bad', `${parity} should be bad`);
    assert.equal(verdictBadge({ verdict: 'alarm', parity }).cls, 'badge bad');
  }
  assert.equal(
    verdictBadge({ verdict: 'alarm', parity: 'alarmed_open' }).label,
    'alarm · escalation open',
  );
  assert.equal(
    verdictBadge({ verdict: 'alarm', parity: 'alarmed_unlinked' }).label,
    'alarm · no escalation',
  );
});

test('verdictBadge: a PARITY_PLAIN state adds nothing to the verdict badge', () => {
  // PARITY_PLAIN is an explicit opt-out, not an oversight bucket: "considered,
  // nothing to add" is a different decision from "no branch for it", and only
  // one of them is safe to leave undocumented.
  for (const parity of PARITY_PLAIN) {
    for (const { verdict, base, cls } of VERDICT_BASES) {
      const out = verdictBadge({ verdict, parity });
      assert.equal(out.label, base, `plain parity '${parity}' must render the bare base`);
      assert.equal(out.cls, cls, `plain parity '${parity}' must keep the verdict's class`);
      assert.ok(!out.label.includes(' · '), `plain parity '${parity}' must add no suffix`);
    }
  }
});

test('verdictBadge: an absent parity renders the bare verdict badge', () => {
  for (const parity of [null, undefined, '']) {
    const out = verdictBadge({ verdict: 'alarm', parity });
    assert.equal(out.label, 'alarm', `parity ${JSON.stringify(parity)} must not be marked`);
    assert.equal(out.cls, 'badge bad');
  }
});

test('verdictBadge: an unrecognised parity is MARKED, not passed through', () => {
  // A non-empty parity in NEITHER declaration is a state this copy of the file
  // has never been told about — the producer added one and an already-open
  // browser is holding a cached bundle. Rendering it as the bare verdict badge
  // would be indistinguishable from a state that DECLINED refinement, i.e. the
  // unknown would fail toward the healthy label.
  for (const { verdict, base } of VERDICT_BASES) {
    const out = verdictBadge({ verdict, parity: 'brand_new_parity' });
    assert.equal(out.label, base + ' · unrecognised parity');
    assert.equal(out.cls, 'badge bad', 'an unknown parity must fail toward "something is wrong"');
  }
});

test('verdictBadge: an inherited Object.prototype member is not a parity state', () => {
  // The table is a plain object literal, so a BARE `PARITY_REFINEMENT[parity]`
  // lookup resolves through Object.prototype: a payload carrying parity
  // 'constructor' / 'toString' / 'valueOf' / 'hasOwnProperty' would find a
  // truthy INHERITED member and render `cls: undefined` with a label ending
  // '· undefined'. The source uses Object.prototype.hasOwnProperty.call for
  // exactly this, and this is the clearest example of behaviour the
  // source-as-text suite could only grep for and never execute.
  for (const parity of ['constructor', 'toString', 'valueOf', 'hasOwnProperty', '__proto__']) {
    const out = verdictBadge({ verdict: 'no_alarm', parity });
    assert.equal(
      out.label,
      'no_alarm · unrecognised parity',
      `parity '${parity}' must take the unrecognised branch`,
    );
    assert.equal(out.cls, 'badge bad', `parity '${parity}' produced cls ${out.cls}`);
    assert.ok(!out.label.endsWith('· undefined'), `parity '${parity}' leaked an inherited member`);
  }
});
