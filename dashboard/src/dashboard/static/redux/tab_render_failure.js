// tab_render_failure.js — decides the text shown in place of a tab whose render
// threw: which tab failed, that the rest of the dashboard is still live, how to
// retry, and what was thrown. app.jsx's TabErrorBoundary only renders it.
//
// LOAD CONTRACT. It reads no window.DF_* global. index.html loads it before the
// Babel JSX tags, so window.DF_TAB_RENDER_FAILURE exists before app.jsx
// destructures it at module scope (test_index_html.py pins the order; datum.js's
// CANONICAL block says why there is no fallback). node requires the same file as
// CommonJS. Coverage is behavioural, in dashboard/tests/js/tab_render_failure.test.mjs.
//
// IT MUST NEVER THROW, whatever was thrown. It runs while React recovers from
// the tab's throw, and a second throw there escapes the boundary and unmounts
// the whole page.

const UNDESCRIBABLE_THROW = 'an undescribable value was thrown';

function nameOfFailedTab(tabLabel) {
  return typeof tabLabel === 'string' && tabLabel !== '' ? tabLabel : 'current';
}

function describeError(error) {
  const name = String(error.name || 'Error');
  const message = String(error.message);
  return message === '' ? name : name + ': ' + message;
}

// Every read of the thrown value sits inside the one try: a getter, a revoked
// proxy or a value String() cannot convert all land on the constant.
function describeThrownValue(thrown) {
  try {
    if (thrown === null || thrown === undefined) return String(thrown) + ' thrown';
    if (typeof thrown === 'function') return 'function thrown';
    if (typeof thrown !== 'object') return typeof thrown + ' thrown: ' + String(thrown);
    if (thrown instanceof Error) return describeError(thrown);
    return 'object thrown, with keys [' + Object.keys(thrown).join(', ') + ']';
  } catch (_undescribable) {
    return UNDESCRIBABLE_THROW;
  }
}

// A function declaration, not a const: app.jsx destructures this name, Babel
// turns that destructure into a global `var`, and a `var` may share the classic
// scripts' scope with a `function` but not with a `const`.
function tabRenderFailureNotice(tabLabel, thrown) {
  return Object.freeze({
    text:
      'The ' + nameOfFailedTab(tabLabel) + ' tab failed to render. ' +
      'The rest of the dashboard is still live; switch tabs to retry this one.',
    detail: describeThrownValue(thrown),
  });
}

// Module-unique export const, never a bare `API` — the CANONICAL note in
// datum.js's header, enforced at runtime by classic_script_scope.test.mjs.
const TAB_RENDER_FAILURE_API = { tabRenderFailureNotice };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = TAB_RENDER_FAILURE_API;
}
if (typeof window !== 'undefined') {
  window.DF_TAB_RENDER_FAILURE = TAB_RENDER_FAILURE_API;
}
