// Shared-classic-script-scope tests for the dashboard's classic (non-module)
// `/static/redux/*.js` <script> tags.
//
// WHY THIS FILE EXISTS — the per-module suites structurally cannot catch this
// bug class. Every other suite in this directory (`graph_layout.test.mjs`,
// `esc_flow_layout.test.mjs`, `prd_grouping.test.mjs`, ...) `import`s exactly
// one module, which resolves as CommonJS (no package.json in this repo) and is
// evaluated in its OWN module scope. In the browser these same files are NOT
// modules: index.html loads each as a classic `<script src=...>` tag, and
// top-level `const`/`let`/`class` bindings in classic scripts all share ONE
// global lexical scope. A name declared by two of them makes the SECOND script
// die at declaration-instantiation time with
// `SyntaxError: Identifier 'X' has already been declared` — before a single
// statement of its body runs, so its trailing `window.DF_* = ...` assignment
// never happens and its consumers see `undefined`. Isolated per-module tests
// are green throughout. This file reconstructs the shared scope with node:vm
// so the collision is visible to CI.
//
// This file is auto-discovered by the `**/*.test.mjs` glob in
// dashboard/tests/test_graph_layout_js.py, which subprocess-runs `node --test`
// over dashboard/tests/js/ — no pytest wrapper change is needed (same as
// esc_flow_layout.test.mjs and runtime_format.test.mjs before it).
//
// DELIBERATE SCOPE LIMIT: only the CLASSIC `.js` scripts are covered. The 15
// `type="text/babel"` `.jsx` tags in index.html are transformed by
// Babel-standalone in the browser at runtime and are not loadable by node:vm
// as-is, so they are out of this harness's reach. None of them currently
// declares a top-level `const API`, but that is not machine-checked here.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

// Mirrors the relative-path idiom in esc_flow_layout.test.mjs /
// runtime_format.test.mjs: resolve the served asset directory from this test
// file's own location rather than from process.cwd().
const REDUX_DIR = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '../../src/dashboard/static/redux',
);

const INDEX_HTML = path.join(REDUX_DIR, 'index.html');

// Each classic script's browser global, keyed by filename. Asserting these are
// defined after the shared load is strictly stronger than "did not throw": it
// proves each script ran all the way through its trailing `window.* =` line.
const EXPECTED_WINDOW_GLOBALS = {
  'data.js': 'DF_DATA_LOADER',
  'graph_layout.js': 'DF_GRAPH_LAYOUT',
  'prd_grouping.js': 'DF_PRD_GROUPING',
  'runtime_format.js': 'DF_RUNTIME_FMT',
  'orch_filter.js': 'DF_ORCH_FILTER',
  'esc_flow_layout.js': 'DF_ESC_FLOW_LAYOUT',
};

function readIndexHtml() {
  return fs.readFileSync(INDEX_HTML, 'utf8');
}

// Matches ONLY the local classic tags — `<script src="/static/redux/<name>.js?v=NN"></script>`.
// Excluded by construction:
//   - the `https://unpkg.com/...` CDN tags (path does not start with /static/redux/);
//   - the `type="text/babel"` .jsx tags (an attribute sits between `<script`
//     and `src`, and the path ends `.jsx`, not `.js`).
const CLASSIC_SCRIPT_RE = /<script\s+src="\/static\/redux\/([A-Za-z0-9_.-]+\.js)(?:\?[^"]*)?"\s*><\/script>/g;

function classicScriptSrcs(html) {
  return [...html.matchAll(CLASSIC_SCRIPT_RE)].map(m => m[1]);
}

// NOTE — no `document` shim, and that omission is load-bearing rather than an
// oversight. data.js:330 gates its polling auto-start on
// `typeof window !== 'undefined' && typeof document !== 'undefined'`, and
// data.js:324-329 documents this exact node/vm affordance: without `document`
// the module stays inert instead of firing real fetches and leaving a live
// setInterval that would hang the test runner. A future maintainer
// "helpfully" adding a `document` shim here would start real polling.
function freshContext() {
  return vm.createContext({ window: {} });
}

// Load every classic script, in index.html's document order, into ONE shared
// context — the browser's actual arrangement. Errors are collected rather than
// thrown so a failure can name every offending script at once.
function loadAllClassicScripts() {
  const ctx = freshContext();
  const srcs = classicScriptSrcs(readIndexHtml());
  const failures = [];

  for (const src of srcs) {
    const source = fs.readFileSync(path.join(REDUX_DIR, src), 'utf8');
    try {
      new vm.Script(source, { filename: src }).runInContext(ctx);
    } catch (err) {
      failures.push({ src, err });
    }
  }

  return { ctx, srcs, failures };
}

function describeFailures(failures) {
  return failures
    .map(({ src, err }) => `  ${src} -> ${err.constructor.name}: ${err.message}`)
    .join('\n');
}

test('index.html exposes the classic /static/redux/*.js scripts in document order', () => {
  const srcs = classicScriptSrcs(readIndexHtml());

  // Zero-match guard. If CLASSIC_SCRIPT_RE silently stops matching (an
  // index.html reformat, a moved asset path, an added attribute), the load
  // test below would iterate zero scripts and pass forever — the same
  // permanent-false-GREEN failure mode test_graph_layout_js.py already guards
  // against with its `# tests N > 0` check.
  assert.ok(
    srcs.length > 0,
    `extracted no classic /static/redux/*.js <script> tags from ${INDEX_HTML} — ` +
      'the extraction regex has gone stale, which would make the shared-scope ' +
      'load test below silently cover nothing',
  );

  for (const expected of ['graph_layout.js', 'esc_flow_layout.js']) {
    assert.ok(
      srcs.includes(expected),
      `expected ${expected} among the extracted classic scripts, got: ${srcs.join(', ')}`,
    );
  }
});

test('every classic /static/redux/*.js script loads into ONE shared global lexical scope without throwing', () => {
  const { failures } = loadAllClassicScripts();

  assert.deepEqual(
    failures.map(f => f.src),
    [],
    'classic script(s) threw when loaded into the shared global lexical scope ' +
      'that index.html actually gives them:\n' +
      `${describeFailures(failures)}\n` +
      'An "Identifier \'X\' has already been declared" SyntaxError means two ' +
      'classic scripts declare the same top-level const/let/class. Classic ' +
      'scripts share ONE global lexical scope, so the second one to declare ' +
      'the name dies before its body runs and never reaches its ' +
      '`window.DF_* = ...` assignment. Fix: give the offending module a ' +
      'module-unique `<MODULE>_API` const, per the existing PRD_GROUPING_API / ' +
      'RUNTIME_FORMAT_API / ORCH_FILTER_API / DF_DATA_LOADER_API precedent.',
  );
});

test('each classic script assigns its window.DF_* global after the shared load', () => {
  const { ctx, srcs } = loadAllClassicScripts();

  for (const src of srcs) {
    const globalName = EXPECTED_WINDOW_GLOBALS[src];
    assert.ok(
      globalName !== undefined,
      `${src} is loaded by index.html but has no entry in EXPECTED_WINDOW_GLOBALS — ` +
        'add its window.DF_* global so this test keeps covering every classic script',
    );

    const consumerNote =
      src === 'esc_flow_layout.js'
        ? ' esc_flow_diagram.jsx:22 does a top-level ' +
          '`const L = window.DF_ESC_FLOW_LAYOUT;`, so an undefined global there ' +
          'makes the escalation-analytics Workflow panel mini-Sankey throw a ' +
          'TypeError on its first L.aggregateFlow(...) call.'
        : '';

    assert.notEqual(
      ctx.window[globalName],
      undefined,
      `${src} did not set window.${globalName} after loading in the shared ` +
        'context — the script aborted before its trailing assignment (a ' +
        'duplicate top-level declaration does exactly this, silently).' +
        consumerNote,
    );
  }
});

test('the harness actually detects a duplicate top-level const (negative control)', () => {
  // Self-proving guard, mirroring
  // test_index_html.py::test_load_order_assertion_fires_on_deferred_cdn.
  // Without it, a harness bug (e.g. accidentally using a fresh context per
  // script) would give a permanent false GREEN once the real collision is
  // fixed. Uses a SEPARATE context so it cannot perturb the tests above.
  const ctx = freshContext();
  const source = fs.readFileSync(path.join(REDUX_DIR, 'graph_layout.js'), 'utf8');
  new vm.Script(source, { filename: 'graph_layout.js' }).runInContext(ctx);

  assert.throws(
    () => {
      new vm.Script('const API = { collide: true };', {
        filename: 'synthetic-collider.js',
      }).runInContext(ctx);
    },
    /has already been declared/,
    'redeclaring graph_layout.js\'s top-level export const in the same context ' +
      'did not throw — the shared-scope harness is not actually sharing scope, ' +
      'so the load test above proves nothing',
  );
});
