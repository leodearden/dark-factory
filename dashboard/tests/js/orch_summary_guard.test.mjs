// Module-contract tests for orch_summary.js — the INTERIM crash guard standing
// between the SPA and the `summary` key that task 5587 removed from
// /api/v2/dashboard/orchestrators. Run via `node --test` (see
// dashboard/tests/test_graph_layout_js.py for the pytest wrapper that surfaces
// this suite in CI via its `**/*.test.mjs` glob — no wrapper change needed for
// this new file).
//
// WHY THIS SUITE EXISTS (esc-5587-4). The defect it guards was not the twenty-
// three unguarded `o.summary.<key>` reads; it was that NOTHING executed them.
// The dashboard suite asserts structural contracts against served .jsx text —
// there is no JS runtime for Babel-transformed files — so a shaper change that
// made the root `App` render body throw on its first real refresh left the
// whole suite green over an SPA that does not mount. Two halves are therefore
// needed and both live here:
//
//   1. BEHAVIOUR — run the actual topbar/tab arithmetic over an ORCHESTRATORS
//      entry with no `summary` key and assert it yields a finite number. That
//      is only possible because the guard is a plain-JS module rather than a
//      helper inside a .jsx, which is the whole reason it is one.
//   2. WIRING — assert no consumer still dereferences `o.summary` directly.
//      Behaviour alone would stay green while a new ungated read reintroduced
//      the crash, and the .jsx sources are reachable here as text even though
//      they are not importable.
//
// orch_summary.js has no package.json in the repo, so it resolves as CommonJS
// (`module.exports = <object>`). Node's cjs-module-lexer cannot statically
// detect named exports assigned from a variable, so
// `import { orchSummary } from '...'` would come back undefined. We therefore
// default-import the module and destructure instead (mirrors
// orch_filter.test.mjs / runtime_format.test.mjs / graph_layout.test.mjs).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

import orchSummaryModule from '../../src/dashboard/static/redux/orch_summary.js';
import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

const {
  ORCH_SUMMARY_KEYS,
  hasOrchSummary,
  orchSummary,
  orchSummaryTotal,
  ORCH_SUMMARY_ABSENT_REASON,
} = orchSummaryModule;

const REDUX_DIR = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '../../src/dashboard/static/redux',
);

// The shape `redux_api.shape_orchestrators` actually puts on the wire after
// task 5587 — transcribed from its projection, `summary` absent BY DESIGN
// (its docstring: a fabricated all-zero summary reads as a measured "this
// orchestrator has no tasks"). This is the fixture the SPA really receives.
const WIRE_ENTRY = {
  pid: 4242,
  pids: [4242],
  label: 'dark-factory',
  project: 'dark-factory',
  project_root: '/home/leo/src/dark-factory',
  running: true,
  started: '2026-09-21T00:00:00Z',
  last_update: null,
  offline: false,
  degraded: false,
};

test('default-imported module exposes the guard functions', () => {
  for (const name of ['hasOrchSummary', 'orchSummary', 'orchSummaryTotal']) {
    assert.equal(
      typeof orchSummaryModule[name],
      'function',
      `orchSummaryModule.${name} should be a function`,
    );
  }
  assert.ok(Array.isArray(ORCH_SUMMARY_KEYS), 'ORCH_SUMMARY_KEYS should be an array');
});

test('module also assigns window.DF_ORCH_SUMMARY (browser dual-export)', () => {
  // The browser half of the dual export. app.jsx / tabs.jsx / tab_overview.jsx
  // destructure this global UNGUARDED at module top level, so an undefined
  // global is the same SPA-fatal failure the guard exists to prevent.
  const source = fs.readFileSync(path.join(REDUX_DIR, 'orch_summary.js'), 'utf8');
  const ctx = { window: {} };
  new Function('window', source)(ctx.window);

  assert.notEqual(ctx.window.DF_ORCH_SUMMARY, undefined);
  assert.equal(typeof ctx.window.DF_ORCH_SUMMARY.orchSummary, 'function');
});

test('an entry with no summary key is reported as not measured', () => {
  assert.equal(hasOrchSummary(WIRE_ENTRY), false);
  assert.equal(hasOrchSummary({ summary: {} }), true);
  // Null is the shape a "measured nothing" producer would most plausibly emit,
  // and `typeof null === 'object'` — so the predicate has to exclude it
  // explicitly or every caller inherits the TypeError it exists to stop.
  assert.equal(hasOrchSummary({ summary: null }), false);
  assert.equal(hasOrchSummary(undefined), false);
});

test('every count key is present and finite for a summary-less entry', () => {
  const counts = orchSummary(WIRE_ENTRY);

  assert.deepEqual(Object.keys(counts).sort(), [...ORCH_SUMMARY_KEYS].sort());
  for (const key of ORCH_SUMMARY_KEYS) {
    assert.equal(
      Number.isFinite(counts[key]),
      true,
      `${key} came back ${counts[key]} — a non-finite value propagates into ` +
        'the reduces and renders "NaN", which is worse than the 0 this guard serves',
    );
    assert.equal(counts[key], 0);
  }
});

test('a measured summary is passed through unchanged', () => {
  const measured = orchSummary({
    ...WIRE_ENTRY,
    summary: { total: 40, done: 31, in_progress: 5, blocked: 2, pending: 2 },
  });

  assert.deepEqual(measured, { total: 40, done: 31, in_progress: 5, blocked: 2, pending: 2 });
});

test('a partially populated summary is completed rather than half-guarded', () => {
  // The producer this guard faces removed a whole key; a producer that drops
  // ONE key is the same class of change, and a `o.summary || {}` guard would
  // pass the hole straight through into the arithmetic.
  const partial = orchSummary({ ...WIRE_ENTRY, summary: { done: 3 } });

  assert.equal(partial.done, 3);
  assert.equal(partial.total, 0);
  assert.equal(partial.blocked, 0);
});

test('non-numeric count values do not reach the arithmetic', () => {
  const coerced = orchSummary({ ...WIRE_ENTRY, summary: { total: '40', done: null, blocked: NaN } });

  assert.equal(coerced.total, 0, 'a string count would concatenate, not add');
  assert.equal(coerced.done, 0);
  assert.equal(coerced.blocked, 0, 'NaN poisons every downstream sum it touches');
});

// ── The computations the guard actually protects ──────────────────────────
// Each mirrors one live call site's arithmetic over the real wire fixture. The
// assertion is deliberately "finite number", not a value: the value is the
// interim zero that task 5589 replaces, while finiteness is the property whose
// loss took the SPA down.

test('app.jsx topbar tasksActive survives a summary-less ORCHESTRATORS list', () => {
  // app.jsx:104, in the ROOT App render body — the fatal one. `DD.ORCHESTRATORS`
  // is seeded `[]`, so the SPA mounts and then throws on the first refresh that
  // populates real entries.
  const orchestrators = [WIRE_ENTRY, { ...WIRE_ENTRY, pid: 4243 }];

  const tasksActive = orchestrators.reduce((n, o) => {
    const s = orchSummary(o);
    return n + s.in_progress + s.blocked;
  }, 0);

  assert.equal(Number.isFinite(tasksActive), true);
  assert.equal(tasksActive, 0);
});

test('tab_overview aggregate reduces survive a summary-less ORCHESTRATORS list', () => {
  const orchestrators = [WIRE_ENTRY];

  for (const key of ORCH_SUMMARY_KEYS) {
    const total = orchestrators.reduce((n, o) => n + orchSummary(o)[key], 0);
    assert.equal(Number.isFinite(total), true, `${key} aggregate went non-finite`);
  }
});

test("the Orchestrators tab's progress bar widths stay finite", () => {
  // tabs.jsx:362-365. `total` is `counts.total || 1`, so the zero shape divides
  // by 1 rather than by 0 — an empty bar, not four `NaN%` width strings that a
  // browser silently discards.
  const counts = orchSummary(WIRE_ENTRY);
  const total = counts.total || 1;

  for (const key of ['done', 'in_progress', 'blocked', 'pending']) {
    const pct = (counts[key] / total) * 100;
    assert.equal(Number.isFinite(pct), true, `${key} width went non-finite`);
    assert.equal(pct, 0);
  }
});

// ── Wiring: no consumer may dereference o.summary directly ────────────────

const CONSUMERS = ['app.jsx', 'tabs.jsx', 'tab_overview.jsx'];

test('no consumer still reads o.summary directly', () => {
  for (const name of CONSUMERS) {
    const source = fs.readFileSync(path.join(REDUX_DIR, name), 'utf8');

    // Deliberately narrow to the `o.summary` binding these three files use for
    // an ORCHESTRATORS entry. `COSTS.summary`, `ESCALATIONS.summary`,
    // `c.summary` and `row.summary` are unrelated payloads on other endpoints
    // and must not be swept in.
    const direct = source.match(/\bo\.summary\b/g) || [];

    assert.deepEqual(
      direct,
      [],
      `${name} dereferences o.summary directly (${direct.length} site(s)). ` +
        'shape_orchestrators does not project that key, so the read throws on ' +
        'the first refresh carrying a real entry. Route it through ' +
        'orchSummary()/hasOrchSummary() from window.DF_ORCH_SUMMARY instead.',
    );
  }
});

test('every consumer destructures the guard it depends on', () => {
  for (const name of CONSUMERS) {
    const source = fs.readFileSync(path.join(REDUX_DIR, name), 'utf8');

    assert.ok(
      /window\.DF_ORCH_SUMMARY/.test(source),
      `${name} calls the guard but never destructures window.DF_ORCH_SUMMARY — ` +
        'the name would resolve to nothing at render time',
    );
  }
});

test('index.html loads orch_summary.js before every consumer', () => {
  // The guard is destructured UNGUARDED at each consumer's module top level
  // (matching the DF_TASK_ROW_CELLS / DF_BURNDOWN_BANDS idiom), so load order
  // is the contract that keeps that destructure from throwing.
  const html = fs.readFileSync(path.join(REDUX_DIR, 'index.html'), 'utf8');
  const guardAt = html.indexOf('/static/redux/orch_summary.js');

  assert.notEqual(guardAt, -1, 'index.html does not load orch_summary.js at all');
  for (const name of CONSUMERS) {
    const consumerAt = html.indexOf(`/static/redux/${name}`);
    assert.notEqual(consumerAt, -1, `index.html does not load ${name}`);
    assert.ok(
      guardAt < consumerAt,
      `index.html loads ${name} before orch_summary.js, so its top-level ` +
        'destructure of window.DF_ORCH_SUMMARY throws and blanks the SPA',
    );
  }
});

// ── A total, or no total: what a Datum tile may be handed ─────────────────
//
// The aggregate tiles and the per-entry pips are γ1 Datum components, and
// plainDatum(<the guard's zero>) renders a confident 0 stamped fresh at
// served_at: the fabricated zero PRD decision 2 forbids, now with false
// provenance on top. orchSummaryTotal answers null instead, and derivedDatum
// turns that null into an unknown Datum with a reason.

const MEASURED_ENTRY = {
  ...WIRE_ENTRY,
  summary: { total: 40, done: 31, in_progress: 5, blocked: 2, pending: 2 },
};

test('(a) orchSummaryTotal sums a key across entries that all measured it', () => {
  assert.equal(orchSummaryTotal([MEASURED_ENTRY, { ...MEASURED_ENTRY, pid: 4243 }], 'in_progress'), 10);
  assert.equal(orchSummaryTotal([MEASURED_ENTRY], 'done'), 31);
});

test('(a) orchSummaryTotal is null when ANY entry lacks a measured summary', () => {
  // A partial sum is an under-count passed off as a total.
  assert.equal(orchSummaryTotal([MEASURED_ENTRY, WIRE_ENTRY], 'in_progress'), null);
  assert.equal(orchSummaryTotal([WIRE_ENTRY], 'done'), null);
  assert.equal(orchSummaryTotal([{ ...WIRE_ENTRY, summary: null }], 'done'), null);
});

test('(a) orchSummaryTotal is null when an entry has no finite value for the key', () => {
  for (const summary of [{ done: 3 }, { in_progress: NaN }, { in_progress: '5' }, { in_progress: null }]) {
    assert.equal(
      orchSummaryTotal([{ ...WIRE_ENTRY, summary }], 'in_progress'),
      null,
      `summary ${JSON.stringify(summary)} was totalled`,
    );
  }
});

test('(a) an empty list totals 0: nothing to measure is a measured nothing', () => {
  assert.equal(orchSummaryTotal([], 'in_progress'), 0);
});

// datum.js loaded for real, through the same window shim index.html's order
// provides: endpoint_staleness.js first, because datum.js destructures its
// formatAge at module scope.
function loadDatum() {
  globalThis.window = { DF_ENDPOINT_STALENESS: staleness };
  const require = createRequire(import.meta.url);
  const specifier = '../../src/dashboard/static/redux/datum.js';
  delete require.cache[require.resolve(specifier)];
  return require(specifier);
}

const ORCHESTRATORS_PATH = '/api/v2/dashboard/orchestrators';

test('(b) an unmeasured total reaches a tile as an unknown Datum carrying the reason', () => {
  const { derivedDatum, datumView, EM_DASH } = loadDatum();
  const receipts = { [ORCHESTRATORS_PATH]: { servedAt: '2026-09-22T12:00:00+00:00', receivedAt: 1000 } };

  const datum = derivedDatum(
    orchSummaryTotal([WIRE_ENTRY], 'in_progress'), ORCHESTRATORS_PATH, ORCH_SUMMARY_ABSENT_REASON, receipts,
  );

  assert.equal(datum.state, 'unknown');
  assert.equal(datum.reason, ORCH_SUMMARY_ABSENT_REASON);
  let formatted = false;
  const view = datumView(datum, { now: 2000, format: () => { formatted = true; return '0'; } });
  assert.equal(view.text, EM_DASH);
  assert.equal(view.title, ORCH_SUMMARY_ABSENT_REASON);
  assert.equal(formatted, false, "a hole must never reach the caller's format");
});

test('(b) before the first /orchestrators receipt the hole says not yet fetched', () => {
  const { derivedDatum } = loadDatum();

  const datum = derivedDatum(
    orchSummaryTotal([WIRE_ENTRY], 'in_progress'), ORCHESTRATORS_PATH, ORCH_SUMMARY_ABSENT_REASON, {},
  );

  assert.equal(datum.state, 'unknown');
  assert.equal(datum.reason, 'not yet fetched');
});

test('(c) the absent reason is declared once, never re-typed by a consumer', () => {
  assert.equal(typeof ORCH_SUMMARY_ABSENT_REASON, 'string', 'orch_summary.js exports no absent reason');
  for (const name of ['tabs.jsx', 'tab_overview.jsx']) {
    const source = fs.readFileSync(path.join(REDUX_DIR, name), 'utf8');
    assert.ok(
      !source.includes(ORCH_SUMMARY_ABSENT_REASON),
      `${name} hand-copies the absent reason; take ORCH_SUMMARY_ABSENT_REASON off window.DF_ORCH_SUMMARY`,
    );
  }
});

// ── Wiring: no Datum component is handed the crash guard's zero ──────────
//
// SOURCE-TEXT LINT, and interim. No suite here can execute a .jsx file,
// because index.html loads Babel from a CDN. So the (b) tests pin what the
// composition renders, and these only pin that the call sites use it. The scan
// is deliberately small: it has no string awareness and sees only one-line
// bindings. It goes with orch_summary.js when leaf γ2 (task 5589) deletes it.

// The `{…}` expression opening at source[openAt], extracted brace-balanced:
// these props nest parens, arrow functions and template literals, so a regex
// to the first `}` would stop inside one.
function balancedFrom(source, openAt) {
  let depth = 0;
  for (let i = openAt; i < source.length; i++) {
    if (source[i] === '{') depth++;
    else if (source[i] === '}' && --depth === 0) return source.slice(openAt + 1, i);
  }
  throw new Error(`unbalanced expression at offset ${openAt}`);
}

function attributeExpressions(source, attribute) {
  const re = new RegExp(`\\b${attribute}=\\{`, 'g');
  return [...source.matchAll(re)].map(m => balancedFrom(source, m.index + m[0].length - 1));
}

// Locals a file binds straight from the crash guard (`const x = …orchSummary(…`
// on one line), read off the source rather than listed here, so a rename
// cannot slip past. They hold the guard's zero whenever /orchestrators did not
// measure the count, which is always, today.
function guardZeroNames(source) {
  return [...source.matchAll(/const\s+(\w+)\s*=[^;\n]*\borchSummary\(/g)].map(m => m[1]);
}

function readsAny(expression, names) {
  return names.filter(name => new RegExp(`\\b${name}\\b`).test(expression));
}

for (const name of ['tabs.jsx', 'tab_overview.jsx']) {
  test(`(d) ${name}: no datum= expression carries the crash guard's zero`, () => {
    const source = fs.readFileSync(path.join(REDUX_DIR, name), 'utf8');
    const tainted = guardZeroNames(source);
    const expressions = attributeExpressions(source, 'datum');
    assert.ok(expressions.length > 0, `found no datum= expression in ${name}`);

    for (const expression of expressions) {
      assert.ok(
        !/\b(?:hasOrchSummary|orchSummary)\(/.test(expression),
        `${name} builds a Datum straight from the crash guard: datum={${expression}}. ` +
          'Hand it derivedDatum(orchSummaryTotal(...), <orchestrators path>, ORCH_SUMMARY_ABSENT_REASON).',
      );
      assert.deepEqual(
        readsAny(expression, tainted),
        [],
        `${name} hands a Datum a local holding the guard's zero: datum={${expression}}`,
      );
    }
  });

  test(`(d) ${name}: the orchestrator task counts go through orchSummaryTotal`, () => {
    const source = fs.readFileSync(path.join(REDUX_DIR, name), 'utf8');
    assert.match(source, /\borchSummaryTotal\(/);
  });
}

test("(e) the Overview 'Active tasks' tile puts no guard zero beside its em-dash", () => {
  const source = fs.readFileSync(path.join(REDUX_DIR, 'tab_overview.jsx'), 'utf8');
  const openAt = source.indexOf('<StatTile label="Active tasks"');
  assert.notEqual(openAt, -1, "tab_overview.jsx has no 'Active tasks' StatTile");
  const closeAt = source.indexOf('/>', openAt);
  const tile = source.slice(openAt, closeAt + 2);

  assert.deepEqual(
    readsAny(tile, guardZeroNames(source)),
    [],
    "the tile's unit or hint still reads the guard's totals, so '/ 0' or '0 done' " +
      'renders beside the em-dash',
  );
  for (const attribute of ['unit', 'hint']) {
    const [expression] = attributeExpressions(tile, attribute);
    assert.ok(expression !== undefined, `the tile has no ${attribute}={…} expression`);
    assert.match(
      expression,
      /[!=]=\s*null/,
      `${attribute}={${expression}} must be omitted when its total is unknown`,
    );
  }
});
