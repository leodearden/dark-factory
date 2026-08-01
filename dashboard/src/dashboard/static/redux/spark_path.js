// spark_path.js — pure scale/path math for the sparkline primitives
// (`Sparkline` and `StepSpark`) rendered by charts.jsx.
//
// This is a plain-JS module: no JSX, no Babel. It is loaded two ways:
//   - In the browser, via a classic `<script src="/static/redux/spark_path.js">`
//     tag (like graph_layout.js/runtime_format.js), which assigns
//     `window.DF_SPARK_PATH`.
//   - In node (no package.json in this repo, so this file resolves as
//     CommonJS), via `require`/`import` for the `node --test` suite under
//     dashboard/tests/js/.
//
// Both export paths are guarded so this file has no effect outside the
// environment it's actually running in.
//
// index.html loads this file (classic script, before the Babel JSX tags) so
// `window.DF_SPARK_PATH` is defined before charts.jsx executes its top-level
// `const { sparkPaths, stepPaths } = window.DF_SPARK_PATH;` destructure.
//
// ── LOAD-BEARING CONTRACT: A HOLE IS NOT A ZERO ────────────────────────────
//
// This file exists to fix a defect (task 3436) in which charts.jsx's Sparkline
// and StepSpark did their scale/path arithmetic inline with no null handling
// at all — `Math.max(...values, 1)`, `Math.min(...values, 0)`, and
// `y = height - ((v - min) / range) * height`. Three distinct failures came
// out of that, all measured:
//
//   1. A `null` hole coerces to 0 in the y expression, so a MISSING sample was
//      drawn as a real point at the value-0 baseline, joined by line segments
//      to both neighbours and indistinguishable from a measured regression to
//      zero. For a proportion series sitting at 0.95 that is a plunge to the
//      floor and back; where the series has a negative value, min < 0 and the
//      fabricated point lands mid-chart instead.
//   2. An `undefined`/`NaN` hole poisoned BOTH extrema to NaN, and
//      `range = NaN - NaN || 1` silently fell back to 1 (NaN being falsy), so
//      min stayed NaN, every y became NaN, and the WHOLE chart rendered
//      nothing.
//   3. An all-hole series still produced max=1/min=0 from the seeds alone and
//      a full flat line along the chart floor — measurements asserted that
//      were never taken.
//
// So: a sample is plotted only if it is a finite number. A hole is carried
// through as `null` at its own index (never compacted, never fabricated) and
// breaks the line into separate subpaths, so nothing is ever drawn across a
// slot that holds no measurement. The behavioural suite is
// dashboard/tests/js/spark_path.test.mjs; charts.jsx's delegation to this
// module is pinned by dashboard/tests/test_charts_null_samples.py.

// ── Is this value a real, plottable measurement? ───────────────────────────
// Finite numbers only. `null`/`undefined` (a missing sample), `NaN`/`±Infinity`
// (garbage that poisons the extrema and blanks the chart) and non-numbers
// (a numeric string would coerce silently through the arithmetic and plot as
// if measured) are all holes. Every consumer's series arrives as JSON numbers,
// so this rejects nothing legitimate.
//
// Crucially it ADMITS an honest 0: distinguishing a measured zero from a
// missing sample is the entire point of this module, and a predicate that also
// rejected 0 would trade the bug for its mirror image. The repo states the
// same invariant in runtime_format.js's rtCell and tab_memory_evals.jsx's
// dash().
function isPlottable(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

// ── Scale a series into plot space ─────────────────────────────────────────
// Returns { min, max, range, stepX, points }.
//
// `points` has the SAME length as `values`: entry i is `null` where values[i]
// is not plottable, else `[x, y]`. x comes from the ORIGINAL index, so a hole
// never shifts the x-axis — compacting instead would silently redate every
// surviving sample.
function sparkScale(values, width, height) {
  const series = values || [];
  const plottable = series.filter(isPlottable);

  // The `1` and `0` seeds are DELIBERATELY PRESERVED from the pre-extraction
  // code — do not "clean them up". They guarantee the axis always spans at
  // least [0, 1], so a flat all-zero series renders at the floor and a tiny
  // series is not amplified into noise. That is the current framing for every
  // sparkline on six tabs; dropping them here would silently restyle all of
  // them, an invisible regression riding along with a defect fix. Only WHICH
  // samples enter the fold changed: holes are excluded, so they can no longer
  // poison an extremum to NaN.
  //
  // (With the seeds retained, a bare `null` could never have moved either
  // extremum anyway — 0 cannot lower a max already seeded with 1, nor raise a
  // min already seeded with 0. The exclusion is load-bearing for
  // undefined/NaN, which did poison both.)
  const max = Math.max(...plottable, 1);
  const min = Math.min(...plottable, 0);
  const range = max - min || 1;
  const stepX = width / Math.max(series.length - 1, 1);

  const points = series.map((v, i) =>
    isPlottable(v) ? [i * stepX, height - ((v - min) / range) * height] : null,
  );

  return { min, max, range, stepX, points };
}

// ── Split scaled points into maximal runs of consecutive real samples ──────
// Each run is the contiguous stretch between holes, and becomes its own SVG
// subpath. This is what makes the line genuinely DISCONTINUOUS across a hole
// rather than drawn straight through it. Module-private: the exported surface
// is deliberately just the four public functions.
function plottableRuns(points) {
  const runs = [];
  let current = null;

  for (const point of points) {
    if (point === null) {
      current = null;
      continue;
    }
    if (current === null) {
      current = [];
      runs.push(current);
    }
    current.push(point);
  }

  return runs;
}

// ── Close one run of line commands into a filled area subpath ──────────────
// Drops to the baseline under the run's own last x, tracks back to its own
// first x, and closes. Takes the points actually DRAWN (which for a lone
// sample is a zero-length segment, and for a single-sample step series is the
// full-width tick).
//
// A run spanning zero horizontal distance contributes NO area: the fill would
// be an invisible zero-width sliver, and the pre-fix single-point behaviour —
// a full-width triangle under one sample — was itself fabricated.
//
// For a hole-free full-width run, x_last is exactly `width` and x_first is
// exactly 0, so this reproduces the pre-fix
// `${linePath} L${width},${height} L0,${height} Z` close character-for-
// character. The per-subpath form is a strict generalisation, not a restyle —
// spark_path.test.mjs pins that by exact string equality.
function closeRunArea(drawn, lineCommands, height) {
  const firstX = drawn[0][0];
  const lastX = drawn[drawn.length - 1][0];
  if (lastX === firstX) return null;
  return `${lineCommands} L${lastX},${height} L${firstX},${height} Z`;
}

// ── Smooth sparkline path builder (charts.jsx's `Sparkline`) ───────────────
// Returns { line, area }, each a space-joined set of subpaths — one per run of
// consecutive real samples. No plottable samples at all yields
// `{ line: '', area: '' }` rather than throwing, so the decline-to-render
// decision stays at the call site, where the component can skip the <svg>
// entirely instead of drawing a synthetic floor-hugging line.
function sparkPaths(values, width, height) {
  const { points } = sparkScale(values, width, height);
  const lines = [];
  const areas = [];

  for (const run of plottableRuns(points)) {
    // A run of one emits a zero-length segment (`M x,y L x,y`) rather than a
    // bare moveto, which renders NOTHING in SVG — the same trap charts.jsx
    // already special-cased for a single data point. Under Sparkline's
    // existing strokeLinecap="round" this shows as a visible dot at the
    // sample's own x, with no synthetic extension into a neighbouring slot
    // that holds no measurement.
    const drawn = run.length === 1 ? [run[0], run[0]] : run;
    const lineCommands = drawn
      .map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`))
      .join(' ');

    lines.push(lineCommands);
    const areaCommands = closeRunArea(drawn, lineCommands, height);
    if (areaCommands !== null) areas.push(areaCommands);
  }

  return { line: lines.join(' '), area: areas.join(' ') };
}

// ── Step path builder (charts.jsx's `StepSpark`) ───────────────────────────
// Horizontal-then-vertical edges, no diagonals, so discrete state transitions
// read as sharp steps. Same run-splitting and per-subpath area rules as
// sparkPaths; only the within-run edge construction differs.
//
// A hole matters MORE here than in the smooth builder: a horizontal edge held
// at the pre-hole y would assert the value PERSISTED through the missing slot,
// and then drop vertically to the fabricated zero — two fabrications per hole
// rather than one.
function stepPaths(values, width, height) {
  const series = values || [];
  const { points } = sparkScale(series, width, height);
  const lines = [];
  const areas = [];

  for (const run of plottableRuns(points)) {
    let drawn;
    let lineCommands;

    if (run.length === 1) {
      // A run of one has no step edge to build, so it is drawn as a single
      // horizontal segment rather than routed through the edge loop below
      // (which emits a horizontal AND a vertical command per point, and would
      // duplicate the `L` here). The pre-extraction code likewise handled its
      // single-data-point case before the edge loop.
      const [x, y] = run[0];

      // Whole series is ONE sample -> keep the documented full-width
      // horizontal tick verbatim (pre-extraction charts.jsx:80-84); without it
      // the path holds only a Move command and renders nothing. No hole is
      // involved, so nothing is asserted about an unmeasured slot.
      //
      // Otherwise this is an ISOLATED sample flanked by holes -> a zero-length
      // segment at its own x, rendering as a visible square under StepSpark's
      // existing strokeLinecap="square". Deliberately NOT extended one step
      // wide: a step held across a known-missing slot would assert the value
      // persisted there, which is the synthetic-data class this module exists
      // to remove.
      const endX = series.length === 1 ? width : x;

      drawn = [run[0], [endX, y]];
      lineCommands = `M${x},${y} L${endX},${y}`;
    } else {
      const parts = [`M${run[0][0]},${run[0][1]}`];
      for (let i = 1; i < run.length; i++) {
        // Horizontal segment to the new x, keeping the old y...
        parts.push(`L${run[i][0]},${run[i - 1][1]}`);
        // ...then the vertical segment to the new y. Sharp step edges.
        parts.push(`L${run[i][0]},${run[i][1]}`);
      }
      drawn = run;
      lineCommands = parts.join(' ');
    }

    lines.push(lineCommands);
    const areaCommands = closeRunArea(drawn, lineCommands, height);
    if (areaCommands !== null) areas.push(areaCommands);
  }

  return { line: lines.join(' '), area: areas.join(' ') };
}

// Module-unique export const, never a bare `API` — see the
// shared-classic-script-scope note in graph_layout.js's header, enforced by
// dashboard/tests/js/classic_script_scope.test.mjs. A collision here would
// leave window.DF_SPARK_PATH undefined and break charts.jsx's top-level
// destructure of it, blanking nearly every tab.
const SPARK_PATH_API = { isPlottable, sparkScale, sparkPaths, stepPaths };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = SPARK_PATH_API;
}
if (typeof window !== 'undefined') {
  window.DF_SPARK_PATH = SPARK_PATH_API;
}
