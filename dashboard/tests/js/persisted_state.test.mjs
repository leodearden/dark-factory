// Module-contract tests for persisted_state.js — the localStorage POLICY for
// dashboard UI preferences, and the two React hooks built on it. The hooks are
// driven through a stand-in for React's useState / useEffect, since the .jsx
// that call them (`type="text/babel"`) are not runnable under node; that those
// files take their hooks from here is pinned in test_persisted_state_consumers.py.
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
// is installed unless asked for: without one is the node path, on which
// nothing runs at load. One asked for is removed again once the module loads.
function loadPersistedState(win = {}, { withDocument = false } = {}) {
  globalThis.window = win;
  if (withDocument) globalThis.document = {};
  try {
    const require = createRequire(import.meta.url);
    delete require.cache[require.resolve(SPECIFIER)];
    return { api: require(SPECIFIER), window: win };
  } finally {
    delete globalThis.document;
  }
}

const { api: persistedApi, window: loadedWindow } = loadPersistedState();
const {
  readPersisted,
  writePersisted,
  prunePersistedBooleans,
  PERSISTED_ENTITY_KEY_PREFIXES,
  createPersistedHooks,
} = persistedApi;

// A Storage double holding string values, recording every call by method name.
// A call for which `failWhen(method, ...args)` is true throws `failure` instead
// of acting.
function fakeStorage(entries = {}, { failWhen = () => false, failure = quotaError() } = {}) {
  const items = new Map(Object.entries(entries));
  const calls = [];
  const act = (method, fn) => (...args) => {
    calls.push([method, ...args]);
    if (failWhen(method, ...args)) throw failure;
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
  assert.deepEqual(
    Object.keys(persistedApi).sort(),
    [
      'PERSISTED_ENTITY_KEY_PREFIXES',
      'createPersistedHooks',
      'prunePersistedBooleans',
      'readPersisted',
      'writePersisted',
    ],
  );
  assert.equal(typeof readPersisted, 'function');
  assert.equal(typeof writePersisted, 'function');
  assert.equal(typeof prunePersistedBooleans, 'function');
  assert.equal(typeof createPersistedHooks, 'function');
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
  const storage = fakeStorage({ k: 'true' }, {
    failWhen: method => method === 'getItem',
    failure: new Error('SecurityError'),
  });
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
    const storage = fakeStorage({ 'df.deps.101': 'true' }, { failWhen: called => called === method, failure });
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

// ── The legacy sweep ────────────────────────────────────────────────────────
//
// Keys the mount-time writes already left in operators' browsers. writePersisted
// stops new growth; this clears what accumulated.

test('PERSISTED_ENTITY_KEY_PREFIXES: the four families measured to grow one key per rendered entity', () => {
  assert.deepEqual(
    [...PERSISTED_ENTITY_KEY_PREFIXES],
    ['df.deps.', 'df.locks.', 'df.curator.files.', 'df.memevals.prov.'],
  );
});

// Every per-entity family in its legacy encoding, a truthy survivor, and keys
// outside the families. 'df.memevals.prov.' was written as '1'/'0' by
// tab_memory_evals.jsx::writeProvOpen; the others as JSON booleans.
function legacyStore(extra = {}, options = {}) {
  return fakeStorage({
    'df.deps.101': 'false',
    'df.locks.102': 'false',
    'df.curator.files.tkt_x': 'false',
    'df.memevals.prov.e1': '0',
    'df.deps.103': 'true',
    'df.open.orch': '{"1":true}',
    'some.other.key': 'false',
    ...extra,
  }, options);
}

const LEGACY_SURVIVORS = ['df.deps.103', 'df.open.orch', 'some.other.key'];

test('prunePersistedBooleans: removes exactly the falsy per-entity keys and returns how many', () => {
  const storage = legacyStore();
  assert.equal(prunePersistedBooleans(PERSISTED_ENTITY_KEY_PREFIXES, storage), 4);
  assert.deepEqual([...storage.items.keys()].sort(), LEGACY_SURVIVORS);
});

test('prunePersistedBooleans: a legacy truthy encoding survives', () => {
  const storage = legacyStore({ 'df.memevals.prov.e2': '1' });
  prunePersistedBooleans(PERSISTED_ENTITY_KEY_PREFIXES, storage);
  assert.equal(storage.getItem('df.memevals.prov.e2'), '1');
});

test('prunePersistedBooleans: a key under a listed prefix holding undecodable garbage is removed too', () => {
  const storage = legacyStore({ 'df.locks.9': '{garbage' });
  assert.equal(prunePersistedBooleans(PERSISTED_ENTITY_KEY_PREFIXES, storage), 5);
  assert.equal(storage.getItem('df.locks.9'), null);
});

test('prunePersistedBooleans: a removeItem that throws does not abort the sweep', () => {
  const storage = legacyStore({}, { failWhen: (method, key) => method === 'removeItem' && key === 'df.deps.101' });
  let removed;
  capturingWarn(() => {
    removed = prunePersistedBooleans(PERSISTED_ENTITY_KEY_PREFIXES, storage);
  });
  assert.equal(removed, 3, 'the three removals that succeeded are counted; the one that threw is not');
  assert.deepEqual([...storage.items.keys()].sort(), ['df.deps.101', ...LEGACY_SURVIVORS].sort());
});

test('prunePersistedBooleans: a null storage removes nothing', () => {
  assert.equal(prunePersistedBooleans(PERSISTED_ENTITY_KEY_PREFIXES, null), 0);
});

test('load: in a browser-like document the module sweeps window.localStorage once', () => {
  const storage = legacyStore();
  loadPersistedState({ localStorage: storage }, { withDocument: true });
  assert.deepEqual([...storage.items.keys()].sort(), LEGACY_SURVIVORS);
});

test('load: with no document (the node path data.js also guards on) the module sweeps nothing', () => {
  const storage = legacyStore();
  loadPersistedState({ localStorage: storage });
  assert.equal(storage.items.size, LEGACY_SURVIVORS.length + 4);
  assert.deepEqual(storage.calls, []);
});

test('load: a throwing localStorage getter neither throws nor withholds window.DF_PERSISTED_STATE', () => {
  const win = {
    get localStorage() { throw new Error('SecurityError: access is denied for this document'); },
  };
  const { api, window: loaded } = loadPersistedState(win, { withDocument: true });
  assert.equal(loaded.DF_PERSISTED_STATE, api);
});

// ── The hooks ───────────────────────────────────────────────────────────────
//
// createPersistedHooks takes React's useState / useEffect, so node drives the
// hooks through `reactStandIn`: enough of React's contract to observe what a
// hook stores across renders.

// State lives in per-call slots, a setter whose value is unchanged schedules
// nothing, and an effect runs after its render only when a dependency changed.
// render() repeats while an effect set state, as React's commit loop does, and
// returns the last render's result.
function reactStandIn() {
  const slots = [];
  let cursor = 0;
  let dirty = false;
  let due = [];
  function useState(initial) {
    const at = cursor++;
    if (!(at in slots)) slots[at] = typeof initial === 'function' ? initial() : initial;
    const setState = next => {
      const value = typeof next === 'function' ? next(slots[at]) : next;
      if (!Object.is(value, slots[at])) {
        slots[at] = value;
        dirty = true;
      }
    };
    return [slots[at], setState];
  }
  function useEffect(effect, deps) {
    const at = cursor++;
    const previous = slots[at];
    if (!previous || deps.some((dep, i) => !Object.is(dep, previous[i]))) {
      slots[at] = deps;
      due.push(effect);
    }
  }
  function render(component) {
    for (let pass = 0; pass < 10; pass += 1) {
      cursor = 0;
      dirty = false;
      due = [];
      const result = component();
      for (const effect of due) effect();
      if (!dirty) return result;
    }
    throw new Error('the hook never settled');
  }
  return { useState, useEffect, render };
}

// The hooks over `storage` as window.localStorage, plus the stand-in's render.
function hooksOver(storage) {
  globalThis.window = { localStorage: storage };
  const react = reactStandIn();
  return { render: react.render, ...createPersistedHooks(react) };
}

const setItemCalls = storage => storage.calls.filter(([method]) => method === 'setItem');

test('usePersistedState: a mount at the default leaves no key behind', () => {
  const storage = fakeStorage();
  const { render, usePersistedState } = hooksOver(storage);
  const [value] = render(() => usePersistedState('df.esc.sort', { key: 'task', dir: 'asc' }));
  assert.deepEqual(value, { key: 'task', dir: 'asc' });
  assert.deepEqual(setItemCalls(storage), []);
  assert.equal(storage.length, 0);
});

test('usePersistedState: the stored value seeds the state at mount', () => {
  const storage = fakeStorage({ 'df.escanalytics.window': '"7d"' });
  const { render, usePersistedState } = hooksOver(storage);
  const [value] = render(() => usePersistedState('df.escanalytics.window', '28d'));
  assert.equal(value, '7d');
});

test('usePersistedState: a value set away from the default is stored, and set back removes the key', () => {
  const storage = fakeStorage();
  const { render, usePersistedState } = hooksOver(storage);
  const mount = () => usePersistedState('df.deps.101', false);
  const [, setExpanded] = render(mount);

  setExpanded(true);
  assert.equal(render(mount)[0], true);
  assert.equal(storage.getItem('df.deps.101'), 'true');

  setExpanded(false);
  render(mount);
  assert.equal(storage.getItem('df.deps.101'), null);
});

test('usePersistedState: a falsy key keeps the state in memory only', () => {
  for (const key of ['', null, undefined]) {
    const storage = fakeStorage();
    const { render, usePersistedState } = hooksOver(storage);
    const mount = () => usePersistedState(key, false);
    const [, setExpanded] = render(mount);
    setExpanded(true);
    assert.equal(render(mount)[0], true, `key ${String(key)}`);
    assert.deepEqual(storage.calls, [], `key ${String(key)}`);
  }
});

test('useOpenSet: a mount at the defaults writes no key', () => {
  const storage = fakeStorage();
  const { render, useOpenSet } = hooksOver(storage);
  const [openMap] = render(() => useOpenSet(['a', 'b'], true, 'df.open.esc'));
  assert.deepEqual(openMap, { a: true, b: true });
  assert.deepEqual(setItemCalls(storage), []);
  assert.equal(storage.length, 0);
});

test('useOpenSet: only the ids toggled away from defaultOpen are stored, and toggling back removes the key', () => {
  const storage = fakeStorage();
  const { render, useOpenSet } = hooksOver(storage);
  const mount = () => useOpenSet(['a', 'b'], true, 'df.open.esc');
  const [, toggle] = render(mount);

  toggle('b');
  assert.deepEqual(render(mount)[0], { a: true, b: false });
  assert.equal(storage.getItem('df.open.esc'), JSON.stringify({ b: false }));

  toggle('b');
  assert.deepEqual(render(mount)[0], { a: true, b: true });
  assert.equal(storage.getItem('df.open.esc'), null);
});

test('useOpenSet: with defaultOpen false, an opened id is the one stored', () => {
  const storage = fakeStorage();
  const { render, useOpenSet } = hooksOver(storage);
  const mount = () => useOpenSet(['a', 'b'], false, 'df.open.esc');
  const [, toggle] = render(mount);
  toggle('a');
  render(mount);
  assert.equal(storage.getItem('df.open.esc'), JSON.stringify({ a: true }));
});

test('useOpenSet: a stored flag seeds its id; every other id opens at defaultOpen', () => {
  const storage = fakeStorage({ 'df.open.esc': JSON.stringify({ b: false }) });
  const { render, useOpenSet } = hooksOver(storage);
  const [openMap] = render(() => useOpenSet(['a', 'b'], true, 'df.open.esc'));
  assert.deepEqual(openMap, { a: true, b: false });
  assert.equal(storage.getItem('df.open.esc'), JSON.stringify({ b: false }));
});

test('useOpenSet: a legacy map holding every id is rewritten to its deviations at mount', () => {
  const storage = fakeStorage({ 'df.open.orch': JSON.stringify({ 1: true, 2: false, 3: true }) });
  const { render, useOpenSet } = hooksOver(storage);
  render(() => useOpenSet(['1', '2', '3'], true, 'df.open.orch'));
  assert.equal(storage.getItem('df.open.orch'), JSON.stringify({ 2: false }));
});

test('useOpenSet: setAll away from the default stores every id; setAll back removes the key', () => {
  const storage = fakeStorage();
  const { render, useOpenSet } = hooksOver(storage);
  const mount = () => useOpenSet(['a', 'b'], true, 'df.open.esc');
  const [, , setAll] = render(mount);

  setAll(false);
  assert.deepEqual(render(mount)[0], { a: false, b: false });
  assert.equal(storage.getItem('df.open.esc'), JSON.stringify({ a: false, b: false }));

  setAll(true);
  render(mount);
  assert.equal(storage.getItem('df.open.esc'), null);
});

test('useOpenSet: ids that arrive after mount open at their stored flag, else at defaultOpen', () => {
  // The payloads start empty and fill on the first poll, so a tab often mounts
  // with no ids. Neither its groups nor the operator's stored flags may be lost.
  const storage = fakeStorage({ 'df.open.esc': JSON.stringify({ b: false }) });
  const { render, useOpenSet } = hooksOver(storage);
  assert.deepEqual(render(() => useOpenSet([], true, 'df.open.esc'))[0], {});

  const [openMap] = render(() => useOpenSet(['a', 'b'], true, 'df.open.esc'));
  assert.deepEqual(openMap, { a: true, b: false });
  assert.equal(storage.getItem('df.open.esc'), JSON.stringify({ b: false }));
});

test('useOpenSet: a stored value that is not a map is ignored rather than thrown on', () => {
  for (const raw of ['5', '"open"', 'null', '[true]']) {
    const storage = fakeStorage({ 'df.open.esc': raw });
    const { render, useOpenSet } = hooksOver(storage);
    const [openMap] = render(() => useOpenSet(['a'], true, 'df.open.esc'));
    assert.deepEqual(openMap, { a: true }, `stored ${raw}`);
  }
});

test('useOpenSet: a falsy storageKey keeps the map in memory only', () => {
  const storage = fakeStorage();
  const { render, useOpenSet } = hooksOver(storage);
  const mount = () => useOpenSet(['a'], true, null);
  const [, toggle] = render(mount);
  toggle('a');
  assert.deepEqual(render(mount)[0], { a: false });
  assert.deepEqual(storage.calls, []);
});
