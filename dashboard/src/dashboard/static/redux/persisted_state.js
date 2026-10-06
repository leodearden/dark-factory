// persisted_state.js — the localStorage POLICY for dashboard UI preferences
// (open/expanded flags, sort and filter choices). It owns no React: the
// usePersistedState / useOpenSet hooks stay in the .jsx and delegate here, so
// the policy has one copy and a storage failure has one place to surface.
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

// Module-unique export const, never a bare `API` — the CANONICAL note in
// datum.js's header, enforced at runtime by classic_script_scope.test.mjs.
const PERSISTED_STATE_API = { readPersisted, writePersisted };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = PERSISTED_STATE_API;
}
if (typeof window !== 'undefined') {
  window.DF_PERSISTED_STATE = PERSISTED_STATE_API;
}
