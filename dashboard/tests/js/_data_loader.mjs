// The data.js loader every data.js node suite shares: a fresh module per
// call, against a shimmed browser-ish global, and a microtask drain.
//
// A helper module, not a test: the `*.test.mjs` glob skips it. Its importers
// are data_poll.test.mjs and data_pause.test.mjs.
//
// A static ESM `import` of data.js is not an option: data.js assigns
// `window.DF_DATA = {...}` at module scope, and an imported module's body runs
// BEFORE the importer's, so `globalThis.window` would still be unset. Hence the
// shim-then-require shape below.
import { createRequire } from 'node:module';

// The object datum.js reads at load; see loadDataJs's comment.
import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/data.js';
const DATUM_MODULE_SPECIFIER = '../../src/dashboard/static/redux/datum.js';

// Loads data.js fresh against a shimmed browser-ish global. Installs
// `globalThis.window` (a bare object recording dispatched events) and a
// counting `globalThis.fetch` BEFORE requiring, then busts the require
// cache so each call re-executes data.js's module body from scratch — the
// module body only runs once per require otherwise, which would leave later
// callers seeing a stale `window.DF_DATA` / stale flow-control singleton
// from a previous test's shim (mirrors runtime_format.test.mjs:33-49).
//
// Deliberately does NOT set `globalThis.document`: index.html loads data.js
// as a classic script where `document` always exists, but every test in
// this file runs under node, so the auto-start guard must see no `document`
// and stay inert, exactly like the real non-browser (node --test)
// environment.
//
// Also wraps `globalThis.setInterval` for the duration of the require and
// clears any interval it captures before returning. This is defensive
// rather than load-bearing for the seamed implementation (which gates
// auto-start on `document` and never calls setInterval here at all), but it
// means step-1's RED run — against pre-seam data.js, which calls
// setInterval unconditionally — cannot hang node --test: the interval is
// recorded (so the "no live timer" assertion still fails honestly) and then
// cleared immediately, regardless of what the caller asserts.
export function loadDataJs({ fetchStub } = {}) {
  const events = [];
  const fetchCalls = [];
  const intervalCalls = [];
  // DF_ENDPOINT_STALENESS is installed for datum.js, which destructures
  // formatAge off it at module scope; datum.js is then loaded through the same
  // shim so it can publish DF_DATUM, which data.js in turn destructures at
  // module scope. index.html gives them exactly this order —
  // endpoint_staleness.js -> datum.js -> data.js — and test_index_html.py pins
  // both edges.
  const win = { dispatchEvent: ev => events.push(ev), DF_ENDPOINT_STALENESS: staleness };
  globalThis.window = win;
  globalThis.fetch = (url, init) => {
    fetchCalls.push({ url, init });
    if (fetchStub) return fetchStub(url, init);
    return Promise.resolve({ ok: true, json: async () => ({}) });
  };

  const require = createRequire(import.meta.url);
  const datumResolved = require.resolve(DATUM_MODULE_SPECIFIER);
  delete require.cache[datumResolved];
  require(DATUM_MODULE_SPECIFIER);
  const resolved = require.resolve(MODULE_SPECIFIER);
  delete require.cache[resolved];

  const originalSetInterval = globalThis.setInterval;
  globalThis.setInterval = (...args) => {
    const handle = originalSetInterval(...args);
    intervalCalls.push(handle);
    return handle;
  };

  let api;
  try {
    api = require(MODULE_SPECIFIER);
  } finally {
    globalThis.setInterval = originalSetInterval;
    for (const handle of intervalCalls) clearInterval(handle);
  }

  return { api, window: win, events, fetchCalls, intervalCalls };
}

// Resolves after the entire pending microtask queue has drained (node
// always fully drains microtasks before running the next macrotask/
// immediate), regardless of how many .then()/await hops a chain needs — so
// awaiting this once after a pollTick() call is enough to let every
// non-held endpoint's fetch -> resp.json() -> applyKey chain run to
// completion before the next tick fires.
export function drain() {
  return new Promise(resolve => setImmediate(resolve));
}
