// persisted_state.js — the localStorage POLICY for dashboard UI preferences
// (open/expanded flags, sort and filter choices), and the one copy of the
// usePersistedState / useOpenSet hooks that apply it. It imports no React:
// createPersistedHooks builds the hooks from the useState / useEffect its
// caller hands in, so node can drive them and a storage failure has one place
// to surface.
//
// THE RULE: a key exists only while its value differs from its default. A
// value back at its default removes the key instead of writing it. The hooks
// this replaces wrote every key on mount, default or not, so storage grew one
// key per rendered entity: measured 3 -> 109 keys on one 53-row Orchestrators
// visit, and 117 -> 217 across four loads in ~15 minutes.
//
// A write failure (QuotaExceededError, a SecurityError where storage is
// disabled) is returned to the caller and reported through console.warn with
// its key. It was previously swallowed by a bare `catch (e) {}`.
//
// LOAD CONTRACT. It reads no window.DF_* global. index.html loads it before the
// Babel JSX tags, so window.DF_PERSISTED_STATE exists before its consumers
// destructure it at module scope (test_index_html.py pins the order). node
// requires the same file as CommonJS. Coverage is behavioural, in
// dashboard/tests/js/persisted_state.test.mjs.
//
// Every exported helper is a function declaration, not a const: a .jsx
// destructure of it compiles to a global `var`, which may share the classic
// scripts' scope with a `function` but not with a `const`.

// The browser's localStorage, or null when there is none. Merely reading
// window.localStorage throws a SecurityError where storage is disabled.
function persistedStateStorage() {
  try {
    return (typeof window !== 'undefined' && window.localStorage) || null;
  } catch (e) {
    return null;
  }
}

// The value stored text encodes, or undefined when there is none or it is not
// JSON. JSON never decodes to undefined, so the two cannot be confused.
function decodePersisted(raw) {
  if (raw === null || raw === undefined) return undefined;
  try {
    return JSON.parse(raw);
  } catch (e) {
    return undefined;
  }
}

function readPersisted(key, defaultValue, storage = persistedStateStorage()) {
  if (!key || !storage) return defaultValue;
  let raw;
  try {
    raw = storage.getItem(key);
  } catch (e) {
    return defaultValue;
  }
  const value = decodePersisted(raw);
  return value === undefined ? defaultValue : value;
}

// Returns { action, error }: action is 'written', 'removed', 'skipped' (no
// key), 'unavailable' (no storage) or 'failed', and error is the thrown error
// on 'failed', otherwise null.
//
// Comparing serialisations is sound HERE, not as a general deep-equal: every
// persisted value is a primitive or is built by spreading its default, so key
// order matches by construction.
function writePersisted(key, value, defaultValue, storage = persistedStateStorage()) {
  if (!key) return { action: 'skipped', error: null };
  if (!storage) return { action: 'unavailable', error: null };
  const isDefault = JSON.stringify(value) === JSON.stringify(defaultValue);
  try {
    if (isDefault) storage.removeItem(key);
    else storage.setItem(key, JSON.stringify(value));
  } catch (error) {
    console.warn(`DF_PERSISTED_STATE: could not persist '${key}'`, error);
    return { action: 'failed', error };
  }
  return { action: isDefault ? 'removed' : 'written', error: null };
}

// The families that grew one key per rendered entity: ChipList's deps and
// locks chips (tabs.jsx, tab_curator.jsx) and tab_memory_evals.jsx's
// provenance toggles. Every one holds a boolean open/expanded flag.
const PERSISTED_ENTITY_KEY_PREFIXES = Object.freeze([
  'df.deps.',
  'df.locks.',
  'df.curator.files.',
  'df.memevals.prov.',
]);

// Removes every key under `prefixes` whose value reads falsy or undecodable,
// and returns how many went. readPersisted decides, so a legacy '1'/'0' reads
// as the number it encodes. Keys are collected before any is removed, because
// a removal reindexes storage.key(i); each removal is guarded on its own, so
// one failure does not end the sweep.
function prunePersistedBooleans(prefixes, storage = persistedStateStorage()) {
  if (!storage) return 0;
  const stale = [];
  for (let i = 0; i < storage.length; i += 1) {
    const key = storage.key(i);
    if (!key || !prefixes.some(prefix => key.startsWith(prefix))) continue;
    if (!readPersisted(key, false, storage)) stale.push(key);
  }
  let removed = 0;
  for (const key of stale) {
    try {
      storage.removeItem(key);
      removed += 1;
    } catch (error) {
      console.warn(`DF_PERSISTED_STATE: could not remove '${key}'`, error);
    }
  }
  return removed;
}

// The id -> open flag map stored under `storageKey`, or {} when there is none
// or the stored value is not such a map.
function storedOpenFlags(storageKey) {
  const stored = readPersisted(storageKey, {});
  return stored && typeof stored === 'object' && !Array.isArray(stored) ? stored : {};
}

// `openMap` plus every id in `ids` it lacks, each at its stored flag, else at
// defaultOpen. Returns `openMap` itself when it lacks none, so a state setter
// handed this sees no change.
function withOpenFlags(openMap, ids, storedFlags, defaultOpen) {
  let added = null;
  for (const id of ids) {
    if (id in openMap) continue;
    if (!added) added = {};
    added[id] = id in storedFlags ? !!storedFlags[id] : defaultOpen;
  }
  return added ? { ...openMap, ...added } : openMap;
}

// The entries of `openMap` that differ from defaultOpen: under THE RULE, the
// only part of an open set that is stored.
function openFlagDeviations(openMap, defaultOpen) {
  return Object.fromEntries(Object.entries(openMap).filter(([, open]) => open !== defaultOpen));
}

// The two preference hooks, built over `react` (React itself, or anything
// carrying its useState and useEffect).
function createPersistedHooks({ useState, useEffect }) {
  // State stored under `storageKey`; a falsy key keeps it in memory only.
  function usePersistedState(storageKey, defaultValue) {
    const [value, setValue] = useState(() => readPersisted(storageKey, defaultValue));
    useEffect(() => { writePersisted(storageKey, value, defaultValue); }, [storageKey, value]);
    return [value, setValue];
  }

  // Open flags for a set of fold groups, keyed by id. An id seen after mount
  // opens at its stored flag, else at defaultOpen: the payloads that supply
  // `ids` start empty and fill on the first poll, so a tab often mounts with
  // none. The stored flags are read once, at mount, for that reason.
  function useOpenSet(ids, defaultOpen = true, storageKey = null) {
    const [storedFlags] = useState(() => storedOpenFlags(storageKey));
    const [openMap, setOpenMap] = useState(() => withOpenFlags({}, ids, storedFlags, defaultOpen));
    const idsKey = ids.join('\0');
    useEffect(() => {
      setOpenMap(m => withOpenFlags(m, ids, storedFlags, defaultOpen));
    }, [idsKey]); // eslint-disable-line react-hooks/exhaustive-deps
    useEffect(() => {
      writePersisted(storageKey, openFlagDeviations(openMap, defaultOpen), {});
    }, [storageKey, openMap, defaultOpen]);
    const toggle = id => setOpenMap(m => ({ ...m, [id]: !m[id] }));
    const setAll = open => setOpenMap(Object.fromEntries(ids.map(id => [id, open])));
    return [openMap, toggle, setAll];
  }

  return { usePersistedState, useOpenSet };
}

// Module-unique export const, never a bare `API` — the CANONICAL note in
// datum.js's header, enforced at runtime by classic_script_scope.test.mjs.
const PERSISTED_STATE_API = {
  readPersisted,
  writePersisted,
  prunePersistedBooleans,
  PERSISTED_ENTITY_KEY_PREFIXES,
  createPersistedHooks,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = PERSISTED_STATE_API;
}
if (typeof window !== 'undefined') {
  window.DF_PERSISTED_STATE = PERSISTED_STATE_API;
}

// Clears the keys the old mount-time writes already left in operators'
// browsers. New growth is prevented by writePersisted, not by this sweep. The
// `document` guard keeps it inert under node, as data.js's polling start is.
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  prunePersistedBooleans(PERSISTED_ENTITY_KEY_PREFIXES);
}
