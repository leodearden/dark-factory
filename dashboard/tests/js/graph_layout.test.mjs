// Module-contract tests for graph_layout.js — a plain-JS (no JSX/Babel)
// Sugiyama ordering-phase module. Run via `node --test` (see
// dashboard/tests/test_graph_layout_js.py for the pytest wrapper that
// surfaces this suite in CI).
//
// graph_layout.js has no package.json in the repo, so it resolves as
// CommonJS (`module.exports = <object>`). Node's cjs-module-lexer cannot
// statically detect named exports assigned from a variable, so
// `import { computeTiers } from '...'` would come back undefined. We
// therefore default-import the module and destructure instead.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import layout from '../../src/dashboard/static/redux/graph_layout.js';

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/graph_layout.js';
const EXPECTED_FUNCTION_NAMES = [
  'computeTiers',
  'partitionComponents',
  'orderRows',
  'countCrossings',
];

test('default-imported module exposes the four layout functions', () => {
  for (const name of EXPECTED_FUNCTION_NAMES) {
    assert.equal(typeof layout[name], 'function', `layout.${name} should be a function`);
  }
});

test('module also assigns window.DF_GRAPH_LAYOUT (browser dual-export)', () => {
  // Shim a bare browser-like global before requiring the module fresh via
  // CommonJS require (a separate module registry from the ESM import
  // above), so the module body's `if (typeof window !== 'undefined')`
  // branch executes against our shim.
  globalThis.window = {};
  try {
    const require = createRequire(import.meta.url);
    const required = require(MODULE_SPECIFIER);

    assert.ok(globalThis.window.DF_GRAPH_LAYOUT, 'window.DF_GRAPH_LAYOUT was not set');

    // The require() and import() calls load the module through two
    // independent module registries (CJS require cache vs. ESM import
    // cache), so the module body executes twice and produces two distinct
    // API object instances. We therefore compare structurally — same set
    // of exported names, each a function — rather than asserting
    // reference/deep equality against the ESM-imported `layout`.
    assert.deepEqual(
      Object.keys(globalThis.window.DF_GRAPH_LAYOUT).sort(),
      EXPECTED_FUNCTION_NAMES.slice().sort(),
    );
    assert.deepEqual(Object.keys(required).sort(), EXPECTED_FUNCTION_NAMES.slice().sort());
    for (const name of EXPECTED_FUNCTION_NAMES) {
      assert.equal(typeof globalThis.window.DF_GRAPH_LAYOUT[name], 'function');
    }
  } finally {
    delete globalThis.window;
  }
});
