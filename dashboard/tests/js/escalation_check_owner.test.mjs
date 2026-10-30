// Module-contract tests for escalation_check_owner.js, which reads a
// dependency_capability escalation's typed `delivered_check_owner` field into
// the flat description the sidebar renders. Its consumer is
// tab_escalations.jsx::EscalationSidebar; that call site is exercised against
// this helper's answers in dashboard/tests/js/jsx_render.test.mjs.
//
// THE TWO TASK IDS DIFFER ON PURPOSE. The record's own task_id is the blocked
// DEPENDENT (only ever re-pended); the field's task_id is the OWNER whose
// declaration sites are edited. A helper that swapped them would pass a
// fixture where they were equal.
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob auto-discovers this file).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const SPECIFIER = '../../src/dashboard/static/redux/escalation_check_owner.js';

function loadEscalationCheckOwner() {
  const win = {};
  globalThis.window = win;
  const require = createRequire(import.meta.url);
  delete require.cache[require.resolve(SPECIFIER)];
  return { api: require(SPECIFIER), window: win };
}

const { api: checkOwnerApi, window: loadedWindow } = loadEscalationCheckOwner();
const { checkOwnerPanel } = checkOwnerApi;

// ── Fixtures ────────────────────────────────────────────────────────────────

const OWNER_TASK = '7514';
const DEPENDENT_TASK = '7600';
const CHECK = 'graph-index-provisioned';
const SIDECAR_PATH = 'plans/x.capabilities.yaml';

function owner(sidecar) {
  return { task_id: OWNER_TASK, check_name: CHECK, sidecar };
}

function sidecar(onMain) {
  return { path: SIDECAR_PATH, on_main: onMain };
}

function row(fields) {
  return {
    id: `esc-${DEPENDENT_TASK}-1`,
    task_id: DEPENDENT_TASK,
    category: 'dependency_capability',
    ...fields,
  };
}

function deepFreeze(value) {
  if (value && typeof value === 'object') {
    for (const child of Object.values(value)) deepFreeze(child);
    Object.freeze(value);
  }
  return value;
}

// ── The module ──────────────────────────────────────────────────────────────

test('exports: the module publishes exactly its API, on module.exports and window.DF_ESCALATION_CHECK_OWNER', () => {
  assert.deepEqual(Object.keys(checkOwnerApi), ['checkOwnerPanel']);
  assert.equal(typeof checkOwnerPanel, 'function');
  // tab_escalations.jsx destructures this global at module scope with no fallback.
  assert.equal(loadedWindow.DF_ESCALATION_CHECK_OWNER, checkOwnerApi);
});

// ── A served owner ──────────────────────────────────────────────────────────

test('a declaring sidecar: the owner comes from the field, the dependent from the row', () => {
  const panel = checkOwnerPanel(row({ delivered_check_owner: owner(sidecar('declares')) }));

  assert.deepEqual(panel, {
    ownerTaskId: OWNER_TASK,
    checkName: CHECK,
    dependentTaskId: DEPENDENT_TASK,
    sidecarPath: SIDECAR_PATH,
    sidecarCaption: 'on main: declares',
    sidecarTone: '',
  });
});

test('no sidecar: the owner and check stand, the sidecar line says there is none', () => {
  const panel = checkOwnerPanel(row({ delivered_check_owner: owner(null) }));

  assert.equal(panel.ownerTaskId, OWNER_TASK);
  assert.equal(panel.checkName, CHECK);
  assert.equal(panel.sidecarPath, null);
  assert.equal(panel.sidecarCaption, 'no sidecar');
  assert.equal(panel.sidecarTone, '');
});

// ── No owner ────────────────────────────────────────────────────────────────

test('no owner object: an absent key, null, a string and an array each give null', () => {
  const absent = row({});
  assert.equal('delivered_check_owner' in absent, false);

  for (const candidate of [absent, row({ delivered_check_owner: null }),
    row({ delivered_check_owner: OWNER_TASK }), row({ delivered_check_owner: [OWNER_TASK] })]) {
    assert.equal(checkOwnerPanel(candidate), null, JSON.stringify(candidate));
  }
});

// ── The on_main arms ────────────────────────────────────────────────────────

test('on_main: only "unverified" is toned; every value, an unknown one included, renders verbatim', () => {
  const cases = [
    ['unverified', 'warn'],
    ['does_not_declare', ''],
    ['next-new-state', ''],
  ];
  for (const [onMain, tone] of cases) {
    const panel = checkOwnerPanel(row({ delivered_check_owner: owner(sidecar(onMain)) }));

    assert.equal(panel.sidecarCaption, `on main: ${onMain}`);
    assert.equal(panel.sidecarTone, tone, onMain);
  }
});

// ── A partial owner ─────────────────────────────────────────────────────────
// A hand-edited queue file, or a writer that leaves a key out, must never
// render as 'T-undefined' or 'on main: undefined'.

test('an owner with no task_id or no check_name names nothing to edit: null', () => {
  const partials = [
    { check_name: CHECK, sidecar: null },
    { task_id: null, check_name: CHECK, sidecar: null },
    { task_id: '', check_name: CHECK, sidecar: null },
    { task_id: OWNER_TASK, sidecar: null },
    { task_id: OWNER_TASK, check_name: '', sidecar: null },
  ];
  for (const partial of partials) {
    assert.equal(checkOwnerPanel(row({ delivered_check_owner: partial })), null, JSON.stringify(partial));
  }
});

test('a sidecar with no on_main says so with the "—" marker, untoned', () => {
  for (const site of [{ path: SIDECAR_PATH }, { path: SIDECAR_PATH, on_main: null }, { path: SIDECAR_PATH, on_main: '' }]) {
    const panel = checkOwnerPanel(row({ delivered_check_owner: owner(site) }));

    assert.equal(panel.sidecarPath, SIDECAR_PATH);
    assert.equal(panel.sidecarCaption, 'on main: —', JSON.stringify(site));
    assert.equal(panel.sidecarTone, '');
  }
});

test('a sidecar with no path is still a sidecar: its path is the "—" marker, never null', () => {
  // The sidebar renders its sidecar branch exactly when sidecarPath is set,
  // so a null here would read as the bare caption of no file at all.
  for (const site of [{ on_main: 'declares' }, { path: null, on_main: 'declares' }, { path: '', on_main: 'declares' }]) {
    const panel = checkOwnerPanel(row({ delivered_check_owner: owner(site) }));

    assert.equal(panel.sidecarPath, '—', JSON.stringify(site));
    assert.equal(panel.sidecarCaption, 'on main: declares');
  }
});

// ── Edge shapes ─────────────────────────────────────────────────────────────

test('a row with no task_id has no dependent; a numeric owner task_id comes back as a string', () => {
  const { task_id: _dropped, ...noTaskId } = row({ delivered_check_owner: owner(null) });
  assert.equal(checkOwnerPanel(noTaskId).dependentTaskId, null);
  assert.equal(checkOwnerPanel({ ...noTaskId, task_id: '' }).dependentTaskId, null);

  const numeric = checkOwnerPanel(row({
    delivered_check_owner: { task_id: 7514, check_name: CHECK, sidecar: null },
  }));
  assert.equal(numeric.ownerTaskId, OWNER_TASK);
});

test('the input row is not mutated', () => {
  const frozen = deepFreeze(row({ delivered_check_owner: owner(sidecar('unverified')) }));
  const before = JSON.stringify(frozen);

  checkOwnerPanel(frozen);

  assert.equal(JSON.stringify(frozen), before);
});
