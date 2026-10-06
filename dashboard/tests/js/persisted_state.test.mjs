// Module-contract tests for persisted_state.js — the localStorage POLICY for
// dashboard UI preferences. The React hooks that use it live in the .jsx
// (`type="text/babel"`, not runnable under node); their delegation is pinned
// structurally in the matching test_tab_*.py / test_chip_list_hooks.py.
//
// The policy: a key exists only while its value differs from the default, and
// a storage failure is reported, never swallowed. Every test drives the module
// through an injected fake Storage carrying only getItem / setItem /
// removeItem / key(i) / length, which is all the module may assume.
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob auto-discovers this file).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const SPECIFIER = '../../src/dashboard/static/redux/persisted_state.js';

// Requires the module fresh against `win` as the browser global. No `document`
// is installed: that is the node path, on which nothing runs at load.
function loadPersistedState(win = {}) {
  globalThis.window = win;
  const require = createRequire(import.meta.url);
  delete require.cache[require.resolve(SPECIFIER)];
  return { api: require(SPECIFIER), window: win };
}

const { api: persistedApi, window: loadedWindow } = loadPersistedState();
const { readPersisted, writePersisted } = persistedApi;

// A Storage double holding string values, recording every call by method name.
// `failOn` names methods that throw `failure` instead of acting.
function fakeStorage(entries = {}, { failOn = [], failure = quotaError() } = {}) {
  const items = new Map(Object.entries(entries));
  const calls = [];
  const act = (method, fn) => (...args) => {
    calls.push([method, ...args]);
    if (failOn.includes(method)) throw failure;
    return fn(...args);
  };
  return {
    items,
    calls,
    getItem: act('getItem', key => (items.has(key) ? items.get(key) : null)),
    setItem: act('setItem', (key, value) => { items.set(key, String(value)); }),
    removeItem: act('removeItem', key => { items.delete(key); }),
    key: act('key', i => [...items.keys()][i] ?? null),
    get length() { return items.size; },
  };
}

// The shape a browser throws when a write exceeds the origin's quota.
function quotaError() {
  const error = new Error('The quota has been exceeded.');
  error.name = 'QuotaExceededError';
  return error;
}

// Replaces console.warn for the duration of `fn`, returning every call's args.
function capturingWarn(fn) {
  const warned = [];
  const original = console.warn;
  console.warn = (...args) => { warned.push(args); };
  try {
    fn();
  } finally {
    console.warn = original;
  }
  return warned;
}

const NOT_WRITTEN = Object.freeze({ error: null });

// ── The module ──────────────────────────────────────────────────────────────

test('exports: the module publishes exactly its API, on module.exports and window.DF_PERSISTED_STATE', () => {
  assert.deepEqual(Object.keys(persistedApi), ['readPersisted', 'writePersisted']);
  assert.equal(typeof readPersisted, 'function');
  assert.equal(typeof writePersisted, 'function');
  // The browser half of the dual export: four .jsx files destructure this
  // global at module scope with no fallback.
  assert.equal(loadedWindow.DF_PERSISTED_STATE, persistedApi);
});

// ── readPersisted ───────────────────────────────────────────────────────────

test('readPersisted: a falsy key returns the default without touching storage', () => {
  const storage = fakeStorage({ '': 'true' });
  for (const key of ['', null, undefined]) {
    assert.equal(readPersisted(key, 'dflt', storage), 'dflt', `key ${String(key)}`);
  }
  assert.deepEqual(storage.calls, []);
});

test('readPersisted: a missing key returns the default', () => {
  assert.deepEqual(readPersisted('df.open.esc', { a: true }, fakeStorage()), { a: true });
});

test('readPersisted: unparseable JSON returns the default', () => {
  assert.equal(readPersisted('df.esc.sort', 'age', fakeStorage({ 'df.esc.sort': '{not json' })), 'age');
});

test('readPersisted: a throwing getItem returns the default and does not throw', () => {
  const storage = fakeStorage({ k: 'true' }, { failOn: ['getItem'], failure: new Error('SecurityError') });
  assert.equal(readPersisted('k', false, storage), false);
});

test('readPersisted: a null storage returns the default', () => {
  assert.equal(readPersisted('k', 7, null), 7);
});

test('readPersisted: otherwise it returns the parsed value', () => {
  const storage = fakeStorage({ flag: 'false', map: '{"101":true}', win: '"7d"' });
  assert.equal(readPersisted('flag', true, storage), false);
  assert.deepEqual(readPersisted('map', {}, storage), { 101: true });
  assert.equal(readPersisted('win', '24h', storage), '7d');
});

// ── writePersisted ──────────────────────────────────────────────────────────

test('writePersisted: a falsy key is skipped and touches nothing', () => {
  const storage = fakeStorage();
  for (const key of ['', null, undefined]) {
    assert.deepEqual(writePersisted(key, true, false, storage), { ...NOT_WRITTEN, action: 'skipped' });
  }
  assert.deepEqual(storage.calls, []);
});

test('writePersisted: a null storage is unavailable', () => {
  assert.deepEqual(writePersisted('k', true, false, null), { ...NOT_WRITTEN, action: 'unavailable' });
});

test('writePersisted: a value equal to its default removes the key rather than writing it', () => {
  const storage = fakeStorage();
  assert.deepEqual(writePersisted('df.deps.101', false, false, storage), { ...NOT_WRITTEN, action: 'removed' });
  assert.deepEqual(
    writePersisted('df.open.esc', { a: true, b: false }, { a: true, b: false }, storage),
    { ...NOT_WRITTEN, action: 'removed' },
  );
  assert.equal(storage.calls.filter(([method]) => method === 'setItem').length, 0);
  assert.equal(storage.length, 0);
});

test('writePersisted: a key holding a non-default value is removed once the value returns to the default', () => {
  const storage = fakeStorage({ 'df.deps.101': 'true', unrelated: '1' });
  assert.equal(writePersisted('df.deps.101', false, false, storage).action, 'removed');
  assert.equal(storage.getItem('df.deps.101'), null);
  assert.equal(storage.getItem('unrelated'), '1');
});

test('writePersisted: a value differing from its default is written as JSON', () => {
  const storage = fakeStorage();
  assert.deepEqual(writePersisted('df.deps.101', true, false, storage), { ...NOT_WRITTEN, action: 'written' });
  assert.deepEqual(
    writePersisted('df.open.esc', { a: true }, {}, storage),
    { ...NOT_WRITTEN, action: 'written' },
  );
  assert.equal(storage.getItem('df.deps.101'), 'true');
  assert.equal(storage.getItem('df.open.esc'), JSON.stringify({ a: true }));
});

for (const [method, value] of [['setItem', true], ['removeItem', false]]) {
  test(`writePersisted: a ${method} quota failure is returned as failed and warned with its key, never swallowed`, () => {
    const failure = quotaError();
    const storage = fakeStorage({ 'df.deps.101': 'true' }, { failOn: [method], failure });
    let result;
    const warned = capturingWarn(() => {
      result = writePersisted('df.deps.101', value, false, storage);
    });
    assert.deepEqual(result, { action: 'failed', error: failure });
    assert.equal(warned.length, 1, 'the failure must reach console.warn exactly once');
    assert.ok(
      warned[0].some(arg => typeof arg === 'string' && arg.includes('df.deps.101')),
      `the warning must name the key; got ${JSON.stringify(warned[0].map(String))}`,
    );
  });
}

// ── The browser's own storage, when none is injected ────────────────────────

test('omitted storage: the module reads and writes window.localStorage', () => {
  const storage = fakeStorage({ 'df.esc.sort': '"age"' });
  globalThis.window = { localStorage: storage };
  assert.equal(readPersisted('df.esc.sort', 'id'), 'age');
  assert.equal(writePersisted('df.esc.filter', 'open', 'all').action, 'written');
  assert.equal(storage.getItem('df.esc.filter'), '"open"');
});

test('omitted storage: an absent window.localStorage reads the default and writes nothing', () => {
  globalThis.window = {};
  assert.equal(readPersisted('k', 7), 7);
  assert.equal(writePersisted('k', 1, 0).action, 'unavailable');
});

test('omitted storage: a localStorage getter that throws (storage disabled) reads the default and never throws', () => {
  globalThis.window = {
    get localStorage() { throw new Error('SecurityError: access is denied for this document'); },
  };
  assert.equal(readPersisted('k', 7), 7);
  assert.equal(writePersisted('k', 1, 0).action, 'unavailable');
});
