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
import {
  burndownBlock,
  orchestratorRow,
  projectRow,
  receiptsFor,
  taskCensus,
  tasksSnapshotEntry,
} from './wire_fixtures.mjs';

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

// ── Decision 2: the burndown bands, legend and parity banner (burndown_bands.js) ──

const BURN_PROPS = { projectFilter: [], displayWindow: '24h' };

// The aggregate and the project are given DIFFERENT sample counts, so a view
// that pairs one block's labels with the other's series cannot look right.
const AGGREGATE_SAMPLES = 5;
const PROJECT_SAMPLES = 3;

function seedBurndown(window, { alarm }) {
  const vocab = window.DF_TASK_VOCAB;
  const verdict = { parity_alarm: alarm, parity_cap: 4, parity_peak: 6, parity_breach_count: 3 };
  seedDashboardData(window, {
    PROJECTS: [projectRow(PROJECT)],
    BURNDOWN: burndownBlock(vocab, AGGREGATE_SAMPLES, { ...verdict, parity_projects: [PROJECT] }),
    BURNDOWN_BY_PROJECT: { [PROJECT]: burndownBlock(vocab, PROJECT_SAMPLES, verdict) },
    __receipt: receiptsFor(window.DF_DATA_LOADER, ['BURNDOWN', 'PROJECTS']),
  });
}

function statusMixPanel(container) {
  const panels = [...container.querySelectorAll('.panel')].filter(panel =>
    panel.querySelector(':scope > .panel-head > .title')?.textContent.startsWith('Status mix'),
  );
  assert.equal(panels.length, 1, 'expected exactly one Status mix panel in the rendered view');
  return panels[0];
}

// Each view: how to reach it, which block it draws, and the projects its call
// site hands parityBannerState.
const BURN_VIEWS = [
  {
    view: 'aggregate',
    show() {},
    block: window => window.DF_DATA.BURNDOWN,
    bannerProjects: window => window.DF_DATA.BURNDOWN.parity_projects,
  },
  {
    view: 'per-project',
    show(window, container) {
      const button = [...container.querySelectorAll('.seg button')].find(b => b.textContent === 'Per project');
      assert.ok(button, 'BurnTab rendered no Per project view switch');
      click(window, button);
    },
    block: window => window.DF_DATA.BURNDOWN_BY_PROJECT[PROJECT],
    bannerProjects: () => null,
  },
];

function mountBurnView(harness, show) {
  const { container } = harness.mount(harness.window.DF_TABS.BurnTab, BURN_PROPS);
  show(harness.window, container);
  return statusMixPanel(container);
}

// A colour as the window's CSSOM serialises a style value: oklch(0.80 ...)
// reads back as oklch(0.8 ...).
function styledColor(window, color) {
  const probe = window.document.createElement('span');
  probe.style.background = color;
  return probe.style.background;
}

function filledPathColors(panel) {
  return [...panel.querySelectorAll('svg path')].map(path => path.getAttribute('fill')).filter(Boolean);
}

for (const { view, show, block, bannerProjects } of BURN_VIEWS) {
  test(`tabs.jsx::BurnTab ${view} view renders one legend entry per burndownLegend item`, t => {
    const harness = openWindow(t);
    seedBurndown(harness.window, { alarm: false });
    const panel = mountBurnView(harness, show);

    const legend = harness.window.DF_BURNDOWN_BANDS.burndownLegend(harness.window.DF_CHARTS.PALETTE);
    for (const { label, color } of legend) {
      const swatchColor = styledColor(harness.window, color);
      const entries = [...panel.querySelectorAll('span')].filter(
        span => span.textContent === label && [...span.children].some(swatch => swatch.style.background === swatchColor),
      );
      assert.equal(entries.length, 1, `the ${view} legend has no single "${label}" entry with its ${color} swatch`);
    }
  });

  test(`tabs.jsx::BurnTab ${view} view draws exactly the bands burndownStacks describes for its own block`, t => {
    const harness = openWindow(t);
    seedBurndown(harness.window, { alarm: false });
    const panel = mountBurnView(harness, show);

    const stacks = harness.window.DF_BURNDOWN_BANDS.burndownStacks(block(harness.window), harness.window.DF_CHARTS.PALETTE);
    assert.deepEqual(
      filledPathColors(panel).sort(),
      // Array.from: a node-realm array, since strict deepEqual compares prototypes.
      Array.from(stacks, stack => stack.color).sort(),
      `the ${view} chart must draw one filled band per burndownStacks entry, in that entry's colour`,
    );

    const { fmtDateTime } = harness.window.DF_SHELL;
    const { BURNDOWN, BURNDOWN_BY_PROJECT } = harness.window.DF_DATA;
    const anySampleLabel = new Set([...BURNDOWN.labels, ...BURNDOWN_BY_PROJECT[PROJECT].labels].map(fmtDateTime));
    const ownLabels = Array.from(block(harness.window).labels, fmtDateTime);
    assert.deepEqual(
      [...panel.querySelectorAll('svg text')].map(text => text.textContent).filter(text => anySampleLabel.has(text)),
      ownLabels,
      `the ${view} chart must lay its bands over its own block's ${ownLabels.length} sample labels`,
    );
  });

  test(`tabs.jsx::BurnTab ${view} view renders the parity banner parityBannerState describes when the alarm is up`, t => {
    const harness = openWindow(t);
    seedBurndown(harness.window, { alarm: true });
    const panel = mountBurnView(harness, show);

    const banner = harness.window.DF_BURNDOWN_BANDS.parityBannerState(block(harness.window), bannerProjects(harness.window));
    assert.ok(banner, 'parityBannerState describes no banner for a raised alarm');
    const rendered = [...panel.querySelectorAll('.badge.bad')];
    assert.equal(rendered.length, 1, `the ${view} view must render exactly one parity banner`);
    const text = rendered[0].textContent;
    assert.ok(
      text.includes(`${banner.peak}`) && text.includes(`${banner.cap}`) && text.endsWith(banner.text),
      `the ${view} banner "${text}" does not carry parityBannerState's peak, cap and text "${banner.text}"`,
    );
  });

  test(`tabs.jsx::BurnTab ${view} view renders no parity banner when the alarm is down, though the chart draws`, t => {
    const harness = openWindow(t);
    seedBurndown(harness.window, { alarm: false });
    const panel = mountBurnView(harness, show);

    assert.equal(harness.window.DF_BURNDOWN_BANDS.parityBannerState(block(harness.window), bannerProjects(harness.window)), null);
    assert.ok(filledPathColors(panel).length > 0, `the ${view} chart drew nothing, so the absent banner proves nothing`);
    assert.deepEqual([...panel.querySelectorAll('.badge.bad')], [], `the ${view} view rendered a parity banner with the alarm down`);
  });
}
