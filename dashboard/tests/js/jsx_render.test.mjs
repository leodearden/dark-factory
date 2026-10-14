// The dashboard's JSX render seam: each render decision extracted into a
// classic-script helper must actually reach the DOM at every call site. The
// real .jsx renders in a jsdom window (dom_harness.mjs); every expected value
// comes from calling the helper itself, never from source text.
//
// All render tests live in this one file: node --test runs each file in its
// own process, and each process pays the Babel compile once.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { buildDashboardWindow, seedDashboardData } from './dom_harness.mjs';
import { orchestratorRow, projectRow, receiptsFor, taskCensus, tasksSnapshotEntry } from './wire_fixtures.mjs';

const PROJECT = 'p';
const TAB_PROPS = { projectFilter: [], search: '' };

function openWindow(t) {
  const harness = buildDashboardWindow();
  t.after(() => harness.close());
  return harness;
}

function click(window, element) {
  window.React.act(() => element.dispatchEvent(new window.MouseEvent('click', { bubbles: true })));
}

function spansDescribedBy(root, descriptor) {
  return [...root.querySelectorAll('span')].filter(
    span =>
      span.title === descriptor.title &&
      span.textContent === descriptor.label &&
      descriptor.cls.split(' ').every(cls => span.classList.contains(cls)),
  );
}

// ── Decision 1: the stranded badge (task_row_cells.js::strandBadgeState) ──

function inProgressTask(stranded) {
  return { id: '42', project: PROJECT, title: 'TT', status: 'in-progress', agent: 'impl', stranded, deps: [] };
}

// One running orchestrator for one project holding one task, as the
// /orchestrators and /tasks polls would land it.
function seedOneTask(window, task) {
  seedDashboardData(window, {
    ORCHESTRATORS: [orchestratorRow({ pid: 1, project: PROJECT, running: true })],
    PROJECTS: [projectRow(PROJECT)],
    TASKS_SNAPSHOT: { [PROJECT]: tasksSnapshotEntry([task], taskCensus({ [task.status]: 1 }, window.DF_TASK_VOCAB)) },
    __receipt: receiptsFor(window.DF_DATA_LOADER, ['ORCHESTRATORS', 'TASKS_SNAPSHOT']),
  });
}

function orchTaskRow(window, container, task) {
  const idText = window.DF_SHELL.taskId(task.id);
  return [...container.querySelectorAll('tbody tr')].find(tr => tr.querySelector('td')?.textContent === idText);
}

function graphNode(container, task) {
  return [...container.querySelectorAll('.node')].find(node => node.querySelector('.ttl')?.textContent === task.title);
}

// Each site's options are the ones its call site passes, and `render` returns
// the element the badge belongs in: the task's row, node or detail pane.
const STRAND_SITES = [
  {
    site: 'tabs.jsx::OrchTab (the task-table agent cell)',
    options: { marginLeft: 4 },
    render(harness, task) {
      const { container } = harness.mount(harness.window.DF_TABS.OrchTab, TAB_PROPS);
      return orchTaskRow(harness.window, container, task);
    },
  },
  {
    site: 'tab_tasks.jsx::TaskGraph renderNode',
    options: { compact: true },
    render(harness, task) {
      const { container } = harness.mount(harness.window.DF_TASKS.TasksTab, TAB_PROPS);
      return graphNode(container, task);
    },
  },
  {
    site: 'tab_tasks.jsx::TaskDetail',
    options: undefined,
    render(harness, task) {
      const { container } = harness.mount(harness.window.DF_TASKS.TasksTab, TAB_PROPS);
      const node = graphNode(container, task);
      assert.ok(node, 'the task never reached the graph, so it cannot be selected');
      click(harness.window, node);
      const detail = container.querySelector('.task-detail');
      const idText = harness.window.DF_SHELL.taskId(task.id);
      return detail?.querySelector('.sub')?.textContent.includes(idText) ? detail : undefined;
    },
  },
];

for (const { site, options, render } of STRAND_SITES) {
  test(`${site} renders the badge strandBadgeState describes for a stranded task`, t => {
    const harness = openWindow(t);
    const task = inProgressTask(true);
    seedOneTask(harness.window, task);
    const element = render(harness, task);
    assert.ok(element, `${site} did not render the task at all`);

    const badge = harness.window.DF_TASK_ROW_CELLS.strandBadgeState(task, options);
    assert.ok(badge, 'strandBadgeState describes no badge for a stranded task');
    assert.equal(
      spansDescribedBy(element, badge).length,
      1,
      `${site} must render exactly one <span> carrying strandBadgeState's title, label and classes`,
    );
  });

  test(`${site} renders no stranded badge for a task that is not stranded, though the task renders`, t => {
    const harness = openWindow(t);
    const task = inProgressTask(false);
    seedOneTask(harness.window, task);
    const element = render(harness, task);
    assert.ok(element, `${site} did not render the task at all, so the absent badge proves nothing`);

    const { strandBadgeState } = harness.window.DF_TASK_ROW_CELLS;
    assert.equal(strandBadgeState(task, options), null);
    const wouldRender = strandBadgeState({ ...task, stranded: true }, options);
    assert.deepEqual(spansDescribedBy(element, wouldRender), [], `${site} rendered a stranded badge for a live task`);
  });
}
