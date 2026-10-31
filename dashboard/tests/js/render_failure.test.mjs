// Module-contract tests for render_failure.js — the text shown in place of
// whatever part of the page threw while rendering: a tab, a chrome region (the
// rail, the status strip, the toolbar) or the whole page. The boundary that
// renders it (shell.jsx::RenderFailureBoundary) is executed in the DOM by
// jsx_render.test.mjs; this suite stays pure-module, so it runs without npm ci.
//
// TOTALITY IS THE CONTRACT. The notice is built while React is recovering from
// a throw; a second throw there escapes the boundary and unmounts the whole
// root — the blank page the boundary exists to prevent. So every case below,
// however hostile the thrown value, must come back as two strings, from every
// describer.
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob auto-discovers this file).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

// Safe as a static import: endpoint_staleness.js touches no browser global at
// load — its window assignment is typeof-guarded.
import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

import { caughtFrom } from './_caught.mjs';

const SPECIFIER = '../../src/dashboard/static/redux/render_failure.js';
const DATUM_SPECIFIER = '../../src/dashboard/static/redux/datum.js';

function freshRequire(specifier, win) {
  globalThis.window = win;
  const require = createRequire(import.meta.url);
  delete require.cache[require.resolve(specifier)];
  return require(specifier);
}

function loadRenderFailure() {
  const win = {};
  return { api: freshRequire(SPECIFIER, win), window: win };
}

// datum.js destructures DF_ENDPOINT_STALENESS at module scope, so the shim
// carries the real API before the require (datum.test.mjs::loadDatumJs).
function loadDatumJs() {
  return freshRequire(DATUM_SPECIFIER, { DF_ENDPOINT_STALENESS: staleness });
}

const { api: failureApi, window: loadedWindow } = loadRenderFailure();
const { tabRenderFailureNotice, chromeRenderFailureNotice, pageRenderFailureNotice } = failureApi;

function assertNoticeShape(notice) {
  assert.equal(typeof notice.text, 'string');
  assert.equal(typeof notice.detail, 'string');
  assert.notEqual(notice.text, '');
  assert.notEqual(notice.detail, '');
}

// ── The module ──────────────────────────────────────────────────────────────

test('exports: the module publishes exactly its API, on module.exports and window.DF_RENDER_FAILURE', () => {
  assert.deepEqual(Object.keys(failureApi), [
    'tabRenderFailureNotice',
    'chromeRenderFailureNotice',
    'pageRenderFailureNotice',
  ]);
  for (const describer of Object.values(failureApi)) assert.equal(typeof describer, 'function');
  // app.jsx destructures this global at module scope with no fallback.
  assert.equal(loadedWindow.DF_RENDER_FAILURE, failureApi);
});

// ── (b) the motivating throw, through the real datum.js ─────────────────────

test('(b) a StatTile handed a bare number names the tab and keeps the TypeError loud, by name and message', () => {
  const { assertDatum } = loadDatumJs();
  const caught = caughtFrom(() => assertDatum(42, 'StatTile'));
  assert.equal(caught.name, 'TypeError');

  const notice = tabRenderFailureNotice('Overview', caught);

  assertNoticeShape(notice);
  assert.ok(notice.text.includes('Overview'), notice.text);
  assert.ok(notice.detail.includes(caught.name), notice.detail);
  assert.ok(notice.detail.includes(caught.message), notice.detail);
});

test('(b) a chrome region that threw is named, the open tab is said to be live, and reload retries', () => {
  const { assertDatum } = loadDatumJs();
  const caught = caughtFrom(() => assertDatum(42, 'DatumReading'));
  assert.equal(caught.name, 'TypeError');

  const notice = chromeRenderFailureNotice('status strip', caught);

  assertNoticeShape(notice);
  assert.ok(notice.text.includes('status strip'), notice.text);
  assert.match(notice.text, /tab is still live/i);
  assert.match(notice.text, /reload/i);
  assert.ok(notice.detail.includes(caught.name), notice.detail);
  assert.ok(notice.detail.includes(caught.message), notice.detail);
});

for (const regionLabel of ['', undefined, null, 42, { region: 'rail' }]) {
  test(`(b) a chrome region labelled ${String(regionLabel) || "''"} still gets a generic, non-empty name`, () => {
    const notice = chromeRenderFailureNotice(regionLabel, new Error('x'));

    assertNoticeShape(notice);
    assert.match(notice.text, /reload/i);
    for (const leaked of ['undefined', 'null', '42', '[object Object]', 'The  ']) {
      assert.ok(!notice.text.includes(leaked), notice.text);
    }
  });
}

test('(b) the whole page that threw says the dashboard failed, and reload retries', () => {
  const { assertDatum } = loadDatumJs();
  const caught = caughtFrom(() => assertDatum(42, 'App'));

  const notice = pageRenderFailureNotice(caught);

  assertNoticeShape(notice);
  assert.match(notice.text, /dashboard failed to render/i);
  assert.match(notice.text, /reload/i);
  assert.ok(notice.detail.includes(caught.name), notice.detail);
  assert.ok(notice.detail.includes(caught.message), notice.detail);
});

// ── (c) Error shapes ────────────────────────────────────────────────────────

test('(c) an Error subclass is named by its own name', () => {
  class DatumError extends Error {
    constructor(message) {
      super(message);
      this.name = 'DatumError';
    }
  }

  const notice = tabRenderFailureNotice('Tasks', new DatumError('no envelope'));

  assert.ok(notice.detail.startsWith('DatumError'), notice.detail);
  assert.ok(notice.detail.includes('no envelope'), notice.detail);
});

test('(c) an Error with an empty message still names its type', () => {
  const notice = tabRenderFailureNotice('Tasks', new RangeError(''));

  assertNoticeShape(notice);
  assert.ok(notice.detail.includes('RangeError'), notice.detail);
});

// ── (d) non-Error throwables ────────────────────────────────────────────────

const NON_ERROR_THROWABLES = [
  ['a string', 'boom', 'boom'],
  ['a number', 42, '42'],
  ['a boolean', false, 'false'],
  ['null', null, 'null'],
  ['undefined', undefined, 'undefined'],
  ['a symbol', Symbol('s'), 'Symbol(s)'],
  ['a bigint', 10n, '10'],
  ['a plain object', { code: 7 }, 'code'],
  ['a null-prototype object', Object.create(null), null],
  ['a function', function thrownFunction() {}, null],
];

// Every describer, with the label its text must carry (null: none to carry).
const DESCRIBERS = [
  ['tab', (thrown) => tabRenderFailureNotice('Merge', thrown), 'Merge'],
  ['chrome', (thrown) => chromeRenderFailureNotice('toolbar', thrown), 'toolbar'],
  ['page', pageRenderFailureNotice, null],
];

for (const [describerName, describe, named] of DESCRIBERS) {
  for (const [label, thrown, mentioned] of NON_ERROR_THROWABLES) {
    test(`(d) ${describerName}: ${label} thrown yields a detail that identifies it`, () => {
      const notice = describe(thrown);

      assertNoticeShape(notice);
      if (named !== null) assert.ok(notice.text.includes(named), notice.text);
      if (mentioned !== null) assert.ok(notice.detail.includes(mentioned), notice.detail);
    });
  }
}

function objectWithKeys(count) {
  return Object.fromEntries(Array.from({ length: count }, (_, i) => ['k' + i, i]));
}

test('(d) a thrown object with many keys lists the first ten and counts the rest', () => {
  const notice = tabRenderFailureNotice('Costs', objectWithKeys(300));

  assertNoticeShape(notice);
  for (let i = 0; i < 10; i += 1) assert.ok(notice.detail.includes('k' + i + ','), notice.detail);
  assert.ok(!notice.detail.includes('k10'), notice.detail);
  assert.ok(notice.detail.includes('…and 290 more'), notice.detail);
});

test('(d) a thrown object with exactly ten keys lists them all, with no remainder', () => {
  const notice = tabRenderFailureNotice('Costs', objectWithKeys(10));

  assert.ok(notice.detail.includes('k9'), notice.detail);
  assert.ok(!notice.detail.includes('more'), notice.detail);
});

// ── (e) totality: the load-bearing invariant ────────────────────────────────

function throwing(what) {
  return () => {
    throw new Error(what + ' is hostile');
  };
}

function errorWithThrowingMessage() {
  const error = new Error('unread');
  Object.defineProperty(error, 'message', { get: throwing('message') });
  return error;
}

function objectWhoseEveryDescriptionThrows() {
  return {
    get name() {
      return throwing('name')();
    },
    get message() {
      return throwing('message')();
    },
    toString: throwing('toString'),
  };
}

function proxyWhoseEveryTrapThrows() {
  return new Proxy(
    {},
    {
      get: throwing('get'),
      has: throwing('has'),
      ownKeys: throwing('ownKeys'),
      getPrototypeOf: throwing('getPrototypeOf'),
      getOwnPropertyDescriptor: throwing('getOwnPropertyDescriptor'),
    },
  );
}

function revokedProxy() {
  const { proxy, revoke } = Proxy.revocable({}, {});
  revoke();
  return proxy;
}

// Object.prototype.valueOf returns the object itself, so String() finds no
// primitive form at all and throws.
function objectWithNoPrimitiveForm() {
  return { toString: () => ({}) };
}

const ADVERSARIAL_THROWABLES = [
  ['an Error whose message getter throws', errorWithThrowingMessage],
  ['an object whose name, message and toString all throw', objectWhoseEveryDescriptionThrows],
  ['a Proxy whose every trap throws', proxyWhoseEveryTrapThrows],
  ['a revoked Proxy', revokedProxy],
  ['an object String() cannot convert', objectWithNoPrimitiveForm],
];

for (const [describerName, describe, named] of DESCRIBERS) {
  for (const [label, make] of ADVERSARIAL_THROWABLES) {
    test(`(e) totality, ${describerName}: ${label} still yields two non-empty strings`, () => {
      const thrown = make();
      let notice;

      assert.doesNotThrow(() => {
        notice = describe(thrown);
      });

      assertNoticeShape(notice);
      if (named !== null) assert.ok(notice.text.includes(named), notice.text);
    });
  }
}

// ── (f) the notice is immutable ─────────────────────────────────────────────

for (const [describerName, describe] of DESCRIBERS) {
  test(`(f) ${describerName}: the returned notice is frozen`, () => {
    assert.ok(Object.isFrozen(describe(new Error('x'))));
  });
}
