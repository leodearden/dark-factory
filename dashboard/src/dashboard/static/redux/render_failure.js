// render_failure.js — decides the text shown in place of whatever part of the
// page threw while rendering: a tab, a chrome region, or the whole page. Each
// notice says what failed, what is still live, how to retry, and what was
// thrown. shell.jsx's RenderFailureBoundary only renders it.
//
// IT MUST NEVER THROW, whatever was thrown. It runs while React recovers from
// the throw, and a second throw there escapes its boundary and unmounts the
// whole page.

const UNDESCRIBABLE_THROW = 'an undescribable value was thrown';
const LISTED_KEY_LIMIT = 10;

function nameOfFailedPart(label, fallback) {
  return typeof label === 'string' && label !== '' ? label : fallback;
}

function describeError(error) {
  const name = String(error.name || 'Error');
  const message = String(error.message);
  return message === '' ? name : name + ': ' + message;
}

function listedKeys(keys) {
  const listed = keys.slice(0, LISTED_KEY_LIMIT).join(', ');
  const unlisted = keys.length - LISTED_KEY_LIMIT;
  return unlisted > 0 ? listed + ', …and ' + unlisted + ' more' : listed;
}

// Every read of the thrown value sits inside the one try: a getter, a revoked
// proxy or a value String() cannot convert all land on the constant.
function describeThrownValue(thrown) {
  try {
    if (thrown === null || thrown === undefined) return String(thrown) + ' thrown';
    if (typeof thrown === 'function') return 'function thrown';
    if (typeof thrown !== 'object') return typeof thrown + ' thrown: ' + String(thrown);
    if (thrown instanceof Error) return describeError(thrown);
    return 'object thrown, with keys [' + listedKeys(Object.keys(thrown)) + ']';
  } catch (_undescribable) {
    return UNDESCRIBABLE_THROW;
  }
}

// Function declarations, not consts: app.jsx destructures these names, Babel
// turns that destructure into a global `var`, and a `var` may share the classic
// scripts' scope with a `function` but not with a `const`.
function tabRenderFailureNotice(tabLabel, thrown) {
  return Object.freeze({
    text:
      'The ' + nameOfFailedPart(tabLabel, 'current') + ' tab failed to render. ' +
      'The rest of the dashboard is still live; switch tabs to retry this one.',
    detail: describeThrownValue(thrown),
  });
}

function chromeRenderFailureNotice(regionLabel, thrown) {
  return Object.freeze({
    text:
      'The ' + nameOfFailedPart(regionLabel, 'page chrome') + ' failed to render. ' +
      'The open tab is still live; reload the page to retry.',
    detail: describeThrownValue(thrown),
  });
}

function pageRenderFailureNotice(thrown) {
  return Object.freeze({
    text: 'The dashboard failed to render. Reload the page to retry.',
    detail: describeThrownValue(thrown),
  });
}

// Module-unique export const, never a bare `API` — the CANONICAL note in
// datum.js's header, enforced at runtime by classic_script_scope.test.mjs.
const RENDER_FAILURE_API = { tabRenderFailureNotice, chromeRenderFailureNotice, pageRenderFailureNotice };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = RENDER_FAILURE_API;
}
if (typeof window !== 'undefined') {
  window.DF_RENDER_FAILURE = RENDER_FAILURE_API;
}
