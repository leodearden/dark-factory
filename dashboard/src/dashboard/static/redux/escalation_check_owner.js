// escalation_check_owner.js — reads a dependency_capability escalation's typed
// `delivered_check_owner` field (escalation/src/escalation/models.py::
// DeliveredCheckOwner) into the flat description the escalation sidebar renders.
//
// The record's own task_id is the blocked DEPENDENT, which is only ever
// re-pended; the field's task_id is the OWNER, whose declaration sites are the
// ones to edit. This module keeps the two apart.
//
// on_main is passed through verbatim. Its per-state prose lives server-side
// (orchestrator/src/orchestrator/delivered_check_owner.py::_SIDECAR_LINE) and
// reaches the reader in row.detail, so the client holds no closed list of
// states; only 'unverified' is toned, because it is the state that changes
// what the reader does next.
//
// A partial owner never renders as 'undefined': an owner naming no task or no
// check gives null, and a sidecar's missing path or on_main shows as '—'.
//
// LOAD CONTRACT. It reads no window.DF_* global. index.html loads it before the
// Babel JSX tags, so window.DF_ESCALATION_CHECK_OWNER exists before
// tab_escalations.jsx destructures it at module scope (test_index_html.py pins
// the order). node requires the same file as CommonJS. Coverage is
// behavioural, in dashboard/tests/js/escalation_check_owner.test.mjs and, at
// the sidebar's call site, dashboard/tests/js/jsx_render.test.mjs.

const NO_CHECK_SIDECAR_CAPTION = 'no sidecar';
const UNVERIFIED_CHECK_SIDECAR = 'unverified';
const MISSING_CHECK_SIDECAR_TEXT = '—';

function plainCheckOwnerObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value : null;
}

function presentCheckOwnerText(value) {
  return value === undefined || value === null || value === '' ? null : String(value);
}

// sidecarPath is null exactly when there is no sidecar: the sidebar renders
// its sidecar branch on it.
function checkSidecarSite(sidecar) {
  if (!sidecar) return { sidecarPath: null, sidecarCaption: NO_CHECK_SIDECAR_CAPTION, sidecarTone: '' };
  const onMain = presentCheckOwnerText(sidecar.on_main);
  return {
    sidecarPath: presentCheckOwnerText(sidecar.path) ?? MISSING_CHECK_SIDECAR_TEXT,
    sidecarCaption: `on main: ${onMain ?? MISSING_CHECK_SIDECAR_TEXT}`,
    sidecarTone: onMain === UNVERIFIED_CHECK_SIDECAR ? 'warn' : '',
  };
}

// A function declaration, not a const: tab_escalations.jsx destructures this
// name, Babel turns that destructure into a global `var`, and a `var` may
// share the classic scripts' scope with a `function` but not with a `const`.
function checkOwnerPanel(row) {
  const owner = plainCheckOwnerObject(row && row.delivered_check_owner);
  const ownerTaskId = owner && presentCheckOwnerText(owner.task_id);
  const checkName = owner && presentCheckOwnerText(owner.check_name);
  if (!ownerTaskId || !checkName) return null;
  return {
    ownerTaskId,
    checkName,
    dependentTaskId: presentCheckOwnerText(row.task_id),
    ...checkSidecarSite(plainCheckOwnerObject(owner.sidecar)),
  };
}

// Module-unique export const, never a bare `API` — the CANONICAL note in
// datum.js's header, enforced at runtime by classic_script_scope.test.mjs.
const ESCALATION_CHECK_OWNER_API = { checkOwnerPanel };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = ESCALATION_CHECK_OWNER_API;
}
if (typeof window !== 'undefined') {
  window.DF_ESCALATION_CHECK_OWNER = ESCALATION_CHECK_OWNER_API;
}
