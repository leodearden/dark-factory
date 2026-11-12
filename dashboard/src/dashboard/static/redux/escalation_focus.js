// escalation_focus.js — resolves a cross-tab focus `{queue, id}` (a memory-eval
// escalation link, handed over by app.jsx) to a row of the ESCALATIONS payload.
//
// It decides only WHICH rows match. tab_escalations.jsx owns what each outcome
// renders: one candidate opens, none shows the miss notice, several show the
// ambiguity notice and open nothing.
//
// LOAD CONTRACT. It reads no window.DF_* global. index.html loads it before the
// Babel JSX tags, so window.DF_ESCALATION_FOCUS exists before
// tab_escalations.jsx destructures it at module scope (test_index_html.py pins
// the order). node requires the same file as CommonJS. Coverage is
// behavioural, in dashboard/tests/js/escalation_focus.test.mjs.

function isUsableFocusPart(value) {
  return typeof value === 'string' && value !== '';
}

// The key is the queue, not the project: an id is unique only within one
// queue, because the queue addresses a record by its filename. The live
// payload carried 17 ids in more than one queue, and 15 (project, id)
// collisions once reconciliation rows began resolving to an owning project.
//
// A function declaration, not a const: tab_escalations.jsx destructures this
// name, Babel turns that destructure into a global `var`, and a `var` may
// share the classic scripts' scope with a `function` but not with a `const`.
function findEscalationRow(escalations, focus) {
  if (!focus || !isUsableFocusPart(focus.queue) || !isUsableFocusPart(focus.id)) {
    return { row: null, candidates: [] };
  }
  const candidates = [];
  for (const sub of (escalations && escalations.subsections) || []) {
    if (sub.id !== focus.queue) continue;
    for (const row of sub.escalations || []) {
      if (row.id === focus.id) candidates.push(row);
    }
  }
  return { row: candidates.length === 1 ? candidates[0] : null, candidates };
}

// Module-unique export const, never a bare `API` — the CANONICAL note in
// datum.js's header, enforced at runtime by classic_script_scope.test.mjs.
const ESCALATION_FOCUS_API = { findEscalationRow };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = ESCALATION_FOCUS_API;
}
if (typeof window !== 'undefined') {
  window.DF_ESCALATION_FOCUS = ESCALATION_FOCUS_API;
}
