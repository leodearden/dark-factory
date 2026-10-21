// The node half of dashboard/tests/test_tab_tasks_deferral.py, its one owner.
// That test runs the REAL collector over _deferral_fixture.py and writes each
// project's TASKS_SNAPSHOT entry, plus the needs-human count Python computes
// with shared.task_deferral.needs_human, to DF_DEFERRAL_RENDER_PAYLOAD. This
// file mounts the real Tasks tab in dom_harness.mjs over that payload. The
// name is off the *.test.mjs glob, so test_graph_layout_js.py never runs it
// without a payload.
//
// Expected texts come from deferral_cell.js itself (the jsx_render.test.mjs
// rule); the expected COUNT comes from Python.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

import { buildDashboardWindow, seedDashboardData } from './dom_harness.mjs';
import { projectRow, receiptsFor, servingEndpoint } from './wire_fixtures.mjs';

const PAYLOAD = JSON.parse(fs.readFileSync(process.env.DF_DEFERRAL_RENDER_PAYLOAD, 'utf8'));
const TAB_PROPS = { projectFilter: [], search: '' };
const CHIP = '[data-testid="tasks-deferral-needs-human"]';

function openWindow(t) {
  const harness = buildDashboardWindow();
  t.after(() => {
    harness.close();
    assert.deepEqual(harness.consoleErrors, [], 'the window wrote to console.error');
  });
  return harness;
}

function mountTasksTab(t, projects) {
  const harness = openWindow(t);
  const loader = harness.window.DF_DATA_LOADER;
  seedDashboardData(harness.window, {
    PROJECTS: Object.keys(projects).map(projectRow),
    TASKS_SNAPSHOT: projects,
    __receipt: {
      ...receiptsFor(loader, ['PROJECTS']),
      [servingEndpoint(loader, 'TASKS_SNAPSHOT').stateKey]: { servedAt: PAYLOAD.served_at, receivedAt: Date.now() },
    },
  });
  const { container, rerender } = harness.mount(harness.window.DF_TASKS.TasksTab, TAB_PROPS);
  return { harness, container, rerender, cells: harness.window.DF_DEFERRAL_CELL };
}

function rowsOf(projects, label) {
  return projects[label].rows.value;
}

function projectPanel(container, label) {
  const head = [...container.querySelectorAll('.proj-head')].find(el => el.querySelector('.name')?.textContent === label);
  return head?.parentElement;
}

function graphNode(root, row) {
  return [...root.querySelectorAll('.node')].find(node => node.querySelector('.ttl')?.textContent === row.title);
}

function click(window, element) {
  window.React.act(() => element.dispatchEvent(new window.MouseEvent('click', { bubbles: true })));
}

function assertDrawsHold(element, cell, where) {
  const holds = element.querySelectorAll('.deferral');
  assert.equal(holds.length, 1, `${where} must draw exactly one .deferral`);
  assert.ok(holds[0].textContent.includes(cell.text), `${where}: ${holds[0].textContent}`);
  if (cell.age !== null) assert.ok(holds[0].textContent.includes(cell.age), `${where}: ${holds[0].textContent}`);
}

test('each graph node draws the hold its row carries, and a row without the key draws none', t => {
  const { container, cells } = mountTasksTab(t, PAYLOAD.projects);
  let drawn = 0;
  for (const label of Object.keys(PAYLOAD.projects)) {
    for (const row of rowsOf(PAYLOAD.projects, label)) {
      const node = graphNode(projectPanel(container, label), row);
      assert.ok(node, `${row.id} never reached the graph`);
      const cell = cells.deferralCellState(row);
      if (cell === null) {
        assert.equal(node.querySelectorAll('.deferral').length, 0, `${row.id} carries no deferral key`);
        continue;
      }
      assertDrawsHold(node, cell, row.id);
      drawn += 1;
    }
  }
  assert.ok(drawn > 0, 'no row carried a hold, so nothing was drawn');
});

test('the unrecorded deferred row reads no deferral recorded', t => {
  const { container } = mountTasksTab(t, PAYLOAD.projects);
  const unrecorded = Object.keys(PAYLOAD.projects).flatMap(label =>
    rowsOf(PAYLOAD.projects, label).filter(row => Object.hasOwn(row, 'deferral') && row.deferral === null),
  );
  assert.equal(unrecorded.length, 1, 'the fixture carries exactly one unrecorded deferred row');
  const node = graphNode(container, unrecorded[0]);
  assert.ok(node.querySelector('.deferral').textContent.includes('no deferral recorded'));
});

test('selecting a deferred node shows the same hold in the detail pane', t => {
  const { harness, container, cells } = mountTasksTab(t, PAYLOAD.projects);
  const [label] = Object.keys(PAYLOAD.projects);
  const held = rowsOf(PAYLOAD.projects, label).filter(row => Object.hasOwn(row, 'deferral'));
  for (const row of [held.find(r => r.deferral !== null), held.find(r => r.deferral === null)]) {
    click(harness.window, graphNode(container, row));
    const detail = container.querySelector('.task-detail');
    assert.ok(detail?.querySelector('h4')?.textContent === row.title, `${row.id} was not selected`);
    assertDrawsHold(detail, cells.deferralCellState(row), `${row.id} detail`);
  }
});

test('each project header counts the holds that need a human, as Python counts them', t => {
  const { container, cells } = mountTasksTab(t, PAYLOAD.projects);
  for (const [label, expected] of Object.entries(PAYLOAD.expected_needs_human)) {
    const chip = projectPanel(container, label).querySelector(`.proj-head ${CHIP}`);
    assert.ok(chip, `${label} shows no needs-human chip`);
    assert.equal(chip.textContent, cells.deferralNeedsHumanChip(rowsOf(PAYLOAD.projects, label)).text);
    assert.ok(chip.textContent.startsWith(`${expected} `), `${label}: ${chip.textContent} is not ${expected}`);
  }
});

function showByPrd(window, container, label) {
  const toggle = [...projectPanel(container, label).querySelectorAll('.proj-head .seg button')]
    .find(button => button.textContent === 'by PRD');
  assert.ok(toggle, `${label} offers no "by PRD" toggle`);
  click(window, toggle);
  assert.ok(projectPanel(container, label).querySelector('.prd-groups'), `${label} did not switch to the by-PRD view`);
}

function agedByADay(projects, label, row) {
  const next = structuredClone(projects);
  const aged = next[label].rows.value.find(r => r.id === row.id);
  aged.deferral.age_seconds += 86400;
  return { next, aged };
}

test('the by-PRD view draws the hold the latest poll carries, not the first one', t => {
  const { harness, container, rerender, cells } = mountTasksTab(t, PAYLOAD.projects);
  const [label] = Object.keys(PAYLOAD.projects);
  showByPrd(harness.window, container, label);
  const row = rowsOf(PAYLOAD.projects, label).find(r => r.deferral && r.deferral.age_seconds !== null);
  assertDrawsHold(graphNode(projectPanel(container, label), row), cells.deferralCellState(row), `${row.id} first poll`);

  const { next, aged } = agedByADay(PAYLOAD.projects, label, row);
  const cell = cells.deferralCellState(aged);
  assert.notEqual(cell.age, cells.deferralCellState(row).age, 'the second poll must change what the age reads');
  seedDashboardData(harness.window, { TASKS_SNAPSHOT: next });
  rerender(TAB_PROPS);

  assertDrawsHold(graphNode(projectPanel(container, label), aged), cell, `${row.id} second poll`);
});

test("an older backend's rows render as before: no hold and no chip", t => {
  const { container } = mountTasksTab(t, PAYLOAD.old_backend);
  assert.ok(container.querySelectorAll('.node').length > 0, 'the rows never rendered, so their absence proves nothing');
  assert.equal(container.querySelectorAll('.deferral').length, 0);
  assert.equal(container.querySelectorAll(CHIP).length, 0);
});
