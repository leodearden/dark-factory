// The dashboard's JSX render seam: each render decision extracted into a
// classic-script helper must actually reach the DOM at every call site. The
// real .jsx renders in a jsdom window (dom_harness.mjs); every expected value
// comes from calling the helper itself, never from source text.
//
// node --test runs each file in its own process, and each process pays the
// Babel compile once, so render tests share as few files as possible: this
// one, and jsx_render_quality.test.mjs for the quality dashboard's seams. The
// window/act/fetch plumbing both use is _render_helpers.mjs.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { caughtFrom } from './_caught.mjs';
import { allowAsyncAct, click, openWindow, settle, stubFetch } from './_render_helpers.mjs';
import { buildDashboardWindow, seedDashboardData } from './dom_harness.mjs';
import {
  burndownBlock,
  escalationAnalytics,
  escalationRow,
  escalationsPayload,
  memoryEvalCard,
  memoryEvalEscalation,
  memoryEvalRow,
  memoryEvalsPayload,
  mergeQueueBlock,
  openEscalationItem,
  orchestratorRow,
  projectRow,
  receiptsFor,
  schedulerLockRow,
  schedulerModule,
  schedulerSnapshot,
  taskCensus,
  taskRow,
  tasksSnapshotEntry,
  wireDatum,
} from './wire_fixtures.mjs';

const PROJECT = 'p';
const TAB_PROPS = { projectFilter: [], search: '' };

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
  return taskRow({ project: PROJECT, agent: 'impl', stranded });
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

// By PRD is the Tasks tab's default view, and it draws no graph until a lanes
// row opens; a test that reads graph nodes lists the project flat first, as a
// user who chose the list would have stored it.
function listFlat(window) {
  window.localStorage.setItem('df.tasksGroupByPrd', JSON.stringify({ [PROJECT]: 'flat' }));
}

// The Tasks tab's detail pane, opened on `task` by clicking its graph node;
// undefined unless the pane names that task.
function taskDetailOf(harness, task) {
  listFlat(harness.window);
  const { container } = harness.mount(harness.window.DF_TASKS.TasksTab, TAB_PROPS);
  const node = graphNode(container, task);
  assert.ok(node, 'the task never reached the graph, so it cannot be selected');
  click(harness.window, node);
  const detail = container.querySelector('.task-detail');
  const idText = harness.window.DF_SHELL.taskId(task.id);
  return detail?.querySelector('.sub')?.textContent.includes(idText) ? detail : undefined;
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
      listFlat(harness.window);
      const { container } = harness.mount(harness.window.DF_TASKS.TasksTab, TAB_PROPS);
      return graphNode(container, task);
    },
  },
  {
    site: 'tab_tasks.jsx::TaskDetail',
    options: undefined,
    render: taskDetailOf,
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
  const verdict = { alarm, cap: 4, peak: 6, breachCount: 3 };
  seedDashboardData(window, {
    PROJECTS: [projectRow(PROJECT)],
    BURNDOWN: burndownBlock(vocab, AGGREGATE_SAMPLES, { ...verdict, projects: [PROJECT] }),
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

// ── Decision 3: the PINNING chip (pins_recovery.js::pinningBadgeState) ──
// pins_recovery is three-state on the wire: true, false, or ABSENT (unknown).
// Only true draws the chip.

const PINNING_ARMS = [
  { arm: 'an item that pins recovery', pinnedTaskIds: ['42'], draws: true },
  { arm: 'an item that pins nothing', pinnedTaskIds: [], draws: false },
  { arm: 'an item with no pins_recovery annotation', pinnedTaskIds: undefined, draws: false },
];

function openItemRow(container, item) {
  return [...container.querySelectorAll('tbody tr')].find(tr => tr.querySelector('td')?.textContent === item.id);
}

for (const { arm, pinnedTaskIds, draws } of PINNING_ARMS) {
  test(`tab_escalation_analytics.jsx open items: ${arm} ${draws ? 'renders the chip pinningBadgeState describes' : 'renders no PINNING chip, though its row renders'}`, t => {
    const harness = openWindow(t);
    const { pinningBadgeState } = harness.window.DF_PINS_RECOVERY;
    const item = openEscalationItem({ id: 'esc-42-1', taskId: '42', pinnedTaskIds });
    assert.equal('pins_recovery' in item, pinnedTaskIds !== undefined, 'the fixture must make absence real, not undefined-valued');
    seedDashboardData(harness.window, {
      ESCALATION_ANALYTICS: escalationAnalytics([{ project: PROJECT, openItems: [item] }]),
      __receipt: receiptsFor(harness.window.DF_DATA_LOADER, ['ESCALATION_ANALYTICS']),
    });
    const { container } = harness.mount(harness.window.DF_TABS.EscalationAnalyticsTab, { projectFilter: [] });

    const row = openItemRow(container, item);
    assert.ok(row, `the open item's row did not render, so the chip assertion would prove nothing`);
    const chip = pinningBadgeState(item);
    if (draws) {
      assert.ok(chip, 'pinningBadgeState describes no chip for an item that pins recovery');
      assert.equal(spansDescribedBy(row, chip).length, 1, 'the row must render exactly one span carrying the chip descriptor');
    } else {
      assert.equal(chip, null);
      const wouldRender = pinningBadgeState(openEscalationItem({ id: item.id, taskId: '42', pinnedTaskIds: ['42'] }));
      assert.deepEqual(spansDescribedBy(row, wouldRender), [], 'the row rendered a PINNING chip for an item that pins nothing');
    }
  });
}

// ── Decision 4: the pinning StatTile (pins_recovery.js::pinningSummary) ──
// Each fixture repeats a task id across two escalations, so the escalation
// count and the distinct pinned-task count differ, and carries one item that
// pins nothing and one with no annotation, neither of which may count.

const PINNING_TILE_FIXTURES = [
  { pinned: 'one distinct pinned task', pins: [['42'], ['42']] },
  { pinned: 'two distinct pinned tasks', pins: [['42'], ['42'], ['43']] },
];

function pinningItems(pins) {
  const pinning = pins.map((pinnedTaskIds, i) => openEscalationItem({ id: `esc-pin-${i}`, taskId: pinnedTaskIds[0], pinnedTaskIds }));
  return [
    ...pinning,
    openEscalationItem({ id: 'esc-none', taskId: '50', pinnedTaskIds: [] }),
    openEscalationItem({ id: 'esc-unknown', taskId: '51' }),
  ];
}

function statTile(container, label) {
  const tiles = [...container.querySelectorAll('.kpi')].filter(
    tile => tile.querySelector('.lbl')?.firstChild?.textContent === label,
  );
  assert.equal(tiles.length, 1, `expected exactly one "${label}" StatTile`);
  return tiles[0];
}

for (const { pinned, pins } of PINNING_TILE_FIXTURES) {
  test(`tab_escalations.jsx::EscalationsTab pinning tile renders pinningSummary's count and ${pinned}`, t => {
    const harness = openWindow(t);
    const items = pinningItems(pins);
    seedDashboardData(harness.window, {
      ESCALATION_ANALYTICS: escalationAnalytics([{ project: PROJECT, openItems: items }]),
      __receipt: receiptsFor(harness.window.DF_DATA_LOADER, ['ESCALATION_ANALYTICS', 'ESCALATIONS']),
    });
    const { container } = harness.mount(harness.window.DF_TABS.EscalationsTab, { projectFilter: [] });

    const { count, pinnedTaskCount } = harness.window.DF_PINS_RECOVERY.pinningSummary(items);
    assert.notEqual(count, pinnedTaskCount, 'the fixture must make escalations and distinct pinned tasks differ');
    const tile = statTile(container, 'pinning');
    assert.equal(tile.querySelector('.val').textContent, String(count), 'the tile value must be pinningSummary().count');

    const hint = tile.querySelector('.lbl > span')?.textContent ?? '';
    const noun = pinnedTaskCount === 1 ? 'task' : 'tasks';
    assert.ok(
      hint.endsWith(` ${pinnedTaskCount} ${noun}`),
      `the hint "${hint}" must count pinningSummary().pinnedTaskCount = ${pinnedTaskCount} ${noun}`,
    );
  });
}

// ── Decision 5: the depth chart's bucket width (merge_queue.js::depthBucketCaption) ──
// The panel names the width the server counted the series at; a series that
// names none names no width.

for (const bucketMinutes of [60, null]) {
  test(`tabs.jsx::MergeTab titles its depth panel with depthBucketCaption for bucket_minutes ${bucketMinutes}`, t => {
    const harness = openWindow(t);
    const block = mergeQueueBlock({ bucketMinutes });
    seedDashboardData(harness.window, {
      MERGE_QUEUE: { [PROJECT]: block },
      __receipt: receiptsFor(harness.window.DF_DATA_LOADER, ['MERGE_QUEUE']),
    });
    const { container } = harness.mount(harness.window.DF_TABS.MergeTab, { projectFilter: [] });

    const titles = [...container.querySelectorAll('.panel-head > .title')]
      .map(title => title.textContent)
      .filter(text => text.startsWith('Merge attempts'));
    const caption = harness.window.DF_MERGE_QUEUE.depthBucketCaption(block.depth.value);
    assert.deepEqual(titles, [`Merge attempts · ${caption}`]);
  });
}

// ── An unread runs.db (merge_queue.js::attemptsOver, worstP95Over, projectAttempts) ──
// A project whose runs.db could not be read renders every runs.db reading as
// unknown: no measured-looking 0, and no chart drawn over nothing.

function mountMergeTab(t, block) {
  const harness = openWindow(t);
  seedDashboardData(harness.window, {
    MERGE_QUEUE: { [PROJECT]: block },
    __receipt: receiptsFor(harness.window.DF_DATA_LOADER, ['MERGE_QUEUE']),
  });
  const { container } = harness.mount(harness.window.DF_TABS.MergeTab, { projectFilter: [] });
  return { harness, container };
}

function panelTitled(container, prefix) {
  const titles = [...container.querySelectorAll('.panel-head > .title')].filter(title => title.textContent.startsWith(prefix));
  assert.equal(titles.length, 1, `expected exactly one panel titled "${prefix}…"`);
  return titles[0].closest('.panel');
}

test('tabs.jsx::MergeTab renders an unread runs.db as unknown, never as measured zeros', t => {
  const { harness, container } = mountMergeTab(t, mergeQueueBlock({ bucketMinutes: 60, unread: true }));
  const DF = harness.window.DF_DATA;
  const { attemptsOver, worstP95Over, projectAttempts } = harness.window.DF_MERGE_QUEUE;
  const { datumView, EM_DASH } = harness.window.DF_DATUM;

  const merges = datumView(attemptsOver(DF, null)).text;
  assert.equal(merges, EM_DASH);
  assert.equal(statTile(container, 'Merges (window)').querySelector('.val').textContent, merges);
  const worst = datumView(worstP95Over(DF, null)).text;
  assert.equal(worst, EM_DASH);
  assert.equal(statTile(container, 'p95 latency · worst project').querySelector('.val').textContent, worst);

  const pips = [...container.querySelectorAll('.pip')].map(pip => pip.textContent);
  assert.ok(pips.includes(`${datumView(projectAttempts(DF, PROJECT)).text} attempts`), JSON.stringify(pips));
  assert.ok(!pips.some(text => text.startsWith('0 attempts')), `an unread project reads a measured 0: ${JSON.stringify(pips)}`);
  for (const prefix of ['Outcomes', 'Merge attempts']) {
    assert.equal(panelTitled(container, prefix).querySelector('svg'), null, `${prefix} draws a chart over an unread window`);
  }
});

test('tabs.jsx::MergeTab counts a measured window\'s attempts with attemptCount', t => {
  const block = mergeQueueBlock({ bucketMinutes: 60, outcomes: { done: 3, conflict: 2 } });
  const { harness, container } = mountMergeTab(t, block);

  const count = harness.window.DF_MERGE_QUEUE.attemptCount(block.attempts.value);
  assert.equal(count, 5);
  assert.equal(statTile(container, 'Merges (window)').querySelector('.val').textContent, String(count));
  for (const prefix of ['Outcomes', 'Merge attempts']) {
    assert.notEqual(panelTitled(container, prefix).querySelector('svg'), null, `${prefix} draws no chart over a measured window`);
  }
});

function withDepthValues(block, values) {
  return { ...block, depth: { ...block.depth, value: { ...block.depth.value, values } } };
}

function panelMeta(container, prefix) {
  return panelTitled(container, prefix).querySelector('.panel-head > .meta').textContent;
}

// The depth series' first bucket is counted whole (test_merge_queue_data.py::test_first_bucket_includes_events_before_cutoff), so its sum is not the window's attempt count; attemptCount is the only one.
for (const values of [[1], [2, 0, 3]]) {
  test(`tabs.jsx::MergeTab reads the depth panel as the series' per-bucket peak over ${JSON.stringify(values)}`, t => {
    const block = withDepthValues(mergeQueueBlock({ bucketMinutes: 60, buckets: values.length }), values);
    const { container } = mountMergeTab(t, block);

    const meta = panelMeta(container, 'Merge attempts');
    assert.ok(meta.startsWith(`peak ${Math.max(0, ...values)}/bucket`), meta);
  });
}

test('tabs.jsx::MergeTab gives a window one attempt count, whatever the depth series sums to', t => {
  const block = withDepthValues(mergeQueueBlock({ bucketMinutes: 60, buckets: 3, outcomes: { done: 1 } }), [1, 0, 1]);
  const { harness, container } = mountMergeTab(t, block);
  const count = harness.window.DF_MERGE_QUEUE.attemptCount(block.attempts.value);

  const pips = [...container.querySelectorAll('.pip')].map(pip => pip.textContent);
  assert.ok(pips.includes(`${count} attempts`), JSON.stringify(pips));
  const outcomes = panelMeta(container, 'Outcomes');
  assert.ok(outcomes.startsWith(`${count} merges`), outcomes);
  assert.equal(statTile(container, 'Merges (window)').querySelector('.val').textContent, String(count));
  const depth = panelMeta(container, 'Merge attempts');
  assert.doesNotMatch(depth, /\battempts\b/);
  assert.doesNotMatch(depth, /\bmerges\b/);
});

// ── The Completion-paths legend (charts.jsx::PALETTE.paths) ──
// Every slice performance.py::get_completion_paths serves draws in a colour of
// its own, so no slice reads as another or as nothing.

const COMPLETION_PATHS = ['one-pass', 'multi-pass', 'via-steward', 'via-interactive', 'requeued', 'blocked'];

// performance.py::PerformanceCards.to_wire, one attempt down each path.
const PERF_CARDS = {
  paths: COMPLETION_PATHS.map(path => ({ path, count: 1, pct: 16.7 })),
  escalation: {
    steward_rate: 0, interactive_rate: 0, interactive_count: 0,
    human_attention: { zero: 0, minimal: 0, significant: 0 },
  },
  hist_outer: { labels: ['0', '1', '2', '3+'], values: [6, 0, 0, 0] },
  hist_inner: { labels: ['0', '1', '2', '3+'], values: [6, 0, 0, 0] },
  ttc: { p50: 1000, p75: 1000, p90: 1000, p95: 1000, count: 2 },
};

function seedPerformance(window) {
  seedDashboardData(window, {
    PROJECTS: [projectRow(PROJECT)],
    PERFORMANCE: {
      [PROJECT]: {
        cards: wireDatum(PERF_CARDS),
        time_centiles_history: { labels: [], p50: [], p95: [] },
        one_pass_history: { labels: [], values: [] },
        escalation_history: { labels: [], values: [] },
      },
    },
    PERFORMANCE_LISTING: wireDatum(1),
    __receipt: receiptsFor(window.DF_DATA_LOADER, ['PERFORMANCE', 'PERFORMANCE_LISTING', 'PROJECTS']),
  });
}

// The 8px swatch beside the legend label that reads *path*.
function legendSwatch(panel, path) {
  const labels = [...panel.querySelectorAll('span')].filter(
    span => span.textContent === path && span.previousElementSibling?.style.width === '8px',
  );
  assert.equal(labels.length, 1, `expected exactly one Completion-paths legend row for "${path}"`);
  return labels[0].previousElementSibling;
}

test('tabs.jsx::PerfTab Completion paths gives every slice its own PALETTE.paths swatch', t => {
  const harness = openWindow(t);
  seedPerformance(harness.window);
  const { container } = harness.mount(harness.window.DF_TABS.PerfTab, { projectFilter: [] });
  const panel = panelTitled(container, 'Completion paths');

  const swatches = COMPLETION_PATHS.map(path => {
    const swatch = legendSwatch(panel, path).style.background;
    assert.notEqual(swatch, '', `the "${path}" slice has no swatch colour`);
    assert.equal(swatch, styledColor(harness.window, harness.window.DF_CHARTS.PALETTE.paths[path]), path);
    return swatch;
  });
  assert.equal(new Set(swatches).size, COMPLETION_PATHS.length, `two slices share a colour: ${JSON.stringify(swatches)}`);
});

// ── Decision 6: a render throw is contained, and its notice is render_failure.js's ──
// React's development build reports every throw a boundary catches on
// console.error: jsdom's report of the replayed throw (its stack, or
// 'undefined' for a non-Error), then a note naming the boundary that caught
// it. So these tests cannot assert an empty console. They assert that every
// report is of a throw RenderFailureBoundary caught, and nothing else.

const CAUGHT_NOTE = 'using the error boundary you provided, ';
const UNCAUGHT_NOTE = 'Consider adding an error boundary';

function assertReportsOnlyContainedThrows(consoleErrors) {
  const caughtNotes = consoleErrors.filter(entry => entry.includes(CAUGHT_NOTE));
  assert.ok(caughtNotes.length > 0, `no throw was reported as caught by a boundary: ${JSON.stringify(consoleErrors)}`);
  for (const note of caughtNotes) {
    assert.ok(note.includes(`${CAUGHT_NOTE}RenderFailureBoundary.`), `a boundary other than RenderFailureBoundary caught: ${note}`);
  }
  for (const entry of consoleErrors) {
    assert.ok(!entry.includes(UNCAUGHT_NOTE), `a throw escaped every boundary: ${entry}`);
    assert.ok(!entry.startsWith('Warning:'), `React warned beside the contained throw: ${entry}`);
  }
}

// Like openWindow, but the test asserts the window's reports itself.
function openContainingWindow(t) {
  const harness = buildDashboardWindow();
  t.after(() => harness.close());
  return harness;
}

const UNIT_NOTICE_TESTID = 'unit-render-failure-banner';
const SIBLING_TESTID = 'boundary-sibling';

function noticeLines(notice) {
  return [...notice.children].map(line => line.textContent);
}

// The boundary around `child`, beside a sibling that is outside it.
function mountInsideBoundary(harness, child) {
  const { React, DF_SHELL, DF_RENDER_FAILURE } = harness.window;
  const describe = thrown => DF_RENDER_FAILURE.tabRenderFailureNotice('Overview', thrown);
  const Page = () =>
    React.createElement(
      'div',
      null,
      React.createElement(DF_SHELL.RenderFailureBoundary, { describe, testId: UNIT_NOTICE_TESTID }, child),
      React.createElement('span', { 'data-testid': SIBLING_TESTID }, 'outside the boundary'),
    );
  const { container } = harness.mount(Page);
  return { container, describe };
}

function assertSiblingRenders(container) {
  assert.equal(container.querySelector(`[data-testid="${SIBLING_TESTID}"]`)?.textContent, 'outside the boundary');
}

test('shell.jsx::RenderFailureBoundary contains a real DatumReading throw and renders its describe() notice', t => {
  const harness = openContainingWindow(t);
  const { React, DF_SHELL, DF_DATUM } = harness.window;
  const { container, describe } = mountInsideBoundary(harness, React.createElement(DF_SHELL.DatumReading, { datum: 42 }));

  const expected = describe(caughtFrom(() => DF_DATUM.datumView(42)));
  const notices = container.querySelectorAll(`[data-testid="${UNIT_NOTICE_TESTID}"]`);
  assert.equal(notices.length, 1, 'the boundary must render exactly one notice in place of the throwing child');
  assert.equal(notices[0].getAttribute('role'), 'alert');
  assert.deepEqual(noticeLines(notices[0]), [expected.text, expected.detail]);
  assertSiblingRenders(container);
  assertReportsOnlyContainedThrows(harness.consoleErrors);
});

test('shell.jsx::RenderFailureBoundary fails over on a thrown null, leaving describing it to describe()', t => {
  const harness = openContainingWindow(t);
  function ThrowsNull() {
    throw null;
  }
  const { container, describe } = mountInsideBoundary(harness, harness.window.React.createElement(ThrowsNull));

  const expected = describe(null);
  const notices = container.querySelectorAll(`[data-testid="${UNIT_NOTICE_TESTID}"]`);
  assert.equal(notices.length, 1, 'a falsy throw must still fail over to the notice');
  assert.deepEqual(noticeLines(notices[0]), [expected.text, expected.detail]);
  assertSiblingRenders(container);
  assertReportsOnlyContainedThrows(harness.consoleErrors);
});

test('shell.jsx::RenderFailureBoundary renders a healthy child unchanged, with no notice', t => {
  const harness = openContainingWindow(t);
  const { React } = harness.window;
  const { container } = mountInsideBoundary(harness, React.createElement('em', { 'data-testid': 'healthy' }, 'fine'));

  assert.equal(container.querySelector('[data-testid="healthy"]')?.textContent, 'fine');
  assert.equal(container.querySelector(`[data-testid="${UNIT_NOTICE_TESTID}"]`), null);
  assertSiblingRenders(container);
  assert.deepEqual(harness.consoleErrors, []);
});

// ── The shell's two Datum readers title a reading with its rule ──
// The non-fresh reason, when datumView gives one, wins over the rule; the rule
// fills the fresh case; with neither, the reading carries no title at all.

const RULE = 'R: how this figure is counted';
const DATUM_READERS = [
  { name: 'DatumReading', outer: container => container.firstElementChild },
  { name: 'Pip', outer: container => container.querySelector('.pip') },
];
const RULED_DATUMS = [
  { arm: 'a fresh Datum', datum: wireDatum(5) },
  { arm: 'a stale Datum', datum: wireDatum(5, { state: 'stale', reason: 'the scheduler stopped answering' }) },
  { arm: 'a lower_bound Datum', datum: wireDatum(5, { state: 'lower_bound', reason: 'the runs ledger starts later' }) },
  { arm: 'an unknown Datum', datum: wireDatum(null, { state: 'unknown', reason: 'the map could not be read' }) },
];

for (const reader of DATUM_READERS) {
  for (const { arm, datum } of RULED_DATUMS) {
    test(`shell.jsx::${reader.name} over ${arm} with a rule titles the reading with datumView's reason, else the rule`, t => {
      const harness = openWindow(t);
      const { DF_SHELL, DF_DATUM } = harness.window;
      const { container } = harness.mount(DF_SHELL[reader.name], { datum, rule: RULE });

      const view = DF_DATUM.datumView(datum, { now: Date.now() });
      const outer = reader.outer(container);
      assert.equal(outer.title, view.title || RULE);
      if (datum.state !== 'fresh') assert.notEqual(outer.title, RULE, 'a non-fresh reason must replace the rule');
      assert.ok(outer.textContent.startsWith(view.text), `the reading must start with datumView's text ${view.text}`);
    });
  }

  test(`shell.jsx::${reader.name} over a fresh Datum with no rule carries no title`, t => {
    const harness = openWindow(t);
    const { container } = harness.mount(harness.window.DF_SHELL[reader.name], { datum: wireDatum(5) });

    assert.equal(reader.outer(container).hasAttribute('title'), false);
  });
}

// ── The page app.jsx bootstraps: each fault stays in its own region ──
// Every fault lands as a malformed payload a poll could deliver, then App's
// own re-render signal; nothing patches a component or writes DF_DATA directly.

const NOTICE_SELECTOR = '[data-testid$="-render-failure-banner"]';

// Lands `payloads` as one poll would: each key with its endpoint's receipt.
function landPayloads(window, payloads) {
  seedDashboardData(window, { ...payloads, __receipt: receiptsFor(window.DF_DATA_LOADER, Object.keys(payloads)) });
}

// The page mounted, after landing `seed` (if any) before the first render.
function openPage(t, seed = null) {
  const harness = openContainingWindow(t);
  if (seed) landPayloads(harness.window, seed);
  return { ...harness, root: harness.mountPage() };
}

// A poll landing on the mounted page, then App's re-render signal.
function landPoll(window, payloads) {
  landPayloads(window, payloads);
  window.React.act(() => window.dispatchEvent(new window.Event('df-data-refresh')));
}

function railButton(root, label) {
  const button = [...root.querySelectorAll('.rail-nav button')].find(candidate =>
    [...candidate.children].some(part => part.textContent === label),
  );
  assert.ok(button, `the rail has no "${label}" button`);
  return button;
}

function openTab(page, label) {
  click(page.window, railButton(page.root, label));
}

function failureNotices(root) {
  return [...root.querySelectorAll(NOTICE_SELECTOR)];
}

function crumb(root) {
  return root.querySelector('.crumbs .here')?.textContent;
}

function assertOneNotice(page, testId, expectedText) {
  const notices = failureNotices(page.root);
  assert.deepEqual(notices.map(notice => notice.dataset.testid), [testId]);
  const [text, detail] = noticeLines(notices[0]);
  assert.equal(text, expectedText);
  assert.ok(
    page.consoleErrors.some(entry => entry.startsWith(detail)),
    `the notice's detail "${detail}" heads none of the reported throws: ${JSON.stringify(page.consoleErrors)}`,
  );
  return detail;
}

function bodyHoldsTabContent(root) {
  const body = root.querySelector('.body');
  return !!body && [...body.children].some(child => !child.matches('[data-testid$="-banner"]'));
}

const CHROME_REGIONS = {
  'navigation rail': root => root.querySelectorAll('.rail-nav button').length > 0,
  'status strip': root => !!root.querySelector('.stat-strip'),
  toolbar: root => {
    const toolbar = root.querySelector('.toolbar');
    return !!toolbar && !toolbar.matches(NOTICE_SELECTOR);
  },
};

function assertChromeRenders(root, except = null) {
  for (const [region, renders] of Object.entries(CHROME_REGIONS)) {
    if (region !== except) assert.ok(renders(root), `the ${region} no longer renders`);
  }
}

test('app.jsx contains a tab throw to the tab, and a fresh boundary retries it after a tab switch', t => {
  const page = openPage(t, { ESCALATION_ANALYTICS: escalationAnalytics([{ project: PROJECT, openItems: [null] }]) });
  const { tabRenderFailureNotice } = page.window.DF_RENDER_FAILURE;

  openTab(page, 'Analytics');
  const detail = assertOneNotice(page, 'tab-render-failure-banner', tabRenderFailureNotice('Analytics', undefined).text);
  assert.ok(detail.startsWith('TypeError'), detail);
  assertChromeRenders(page.root);

  openTab(page, 'Costs');
  assert.deepEqual(failureNotices(page.root), []);
  assert.equal(crumb(page.root), 'Costs');

  landPoll(page.window, { ESCALATION_ANALYTICS: escalationAnalytics([{ project: PROJECT, openItems: [] }]) });
  openTab(page, 'Analytics');
  assert.deepEqual(failureNotices(page.root), [], 'the fixed tab still shows the notice: no fresh boundary retried it');
  assert.ok(bodyHoldsTabContent(page.root), 'the fixed tab rendered no content');
  assertReportsOnlyContainedThrows(page.consoleErrors);
});

// One fault per chrome region, each a payload a poll could land, and each
// reaching only that region's render on the Costs tab, which reads none of
// the faulted keys.
const CHROME_FAULTS = [
  {
    region: 'navigation rail',
    testId: 'rail-render-failure-banner',
    // An object where the pending count belongs.
    payloads: () => ({ ESCALATIONS: { summary: { by_status: { pending: { count: 3 } } } } }),
  },
  {
    region: 'status strip',
    testId: 'status-strip-render-failure-banner',
    // A fresh Datum with no value, which datum.py forbids: the topbar's write
    // queue reading then throws inside DatumReading.
    payloads: () => ({ MEMORY_STATUS: { queue: { stats: wireDatum(null) } } }),
  },
  {
    region: 'toolbar',
    testId: 'toolbar-render-failure-banner',
    payloads: () => ({ PROJECTS: [null] }),
  },
];

for (const { region, testId, payloads } of CHROME_FAULTS) {
  test(`app.jsx contains a ${region} throw to the ${region}, and every other region renders on`, t => {
    const page = openPage(t);
    openTab(page, 'Costs');

    landPoll(page.window, payloads());

    const { chromeRenderFailureNotice } = page.window.DF_RENDER_FAILURE;
    assertOneNotice(page, testId, chromeRenderFailureNotice(region, undefined).text);
    assert.equal(crumb(page.root), 'Costs');
    assert.ok(bodyHoldsTabContent(page.root), 'the open tab stopped rendering');
    assertChromeRenders(page.root, region);
    assertReportsOnlyContainedThrows(page.consoleErrors);
  });
}

test("app.jsx leaves a full-page notice naming the error when App's own body throws, not an empty #root", t => {
  const harness = openContainingWindow(t);
  // A malformed /orchestrators payload: App's body filters it while deriving
  // the topbar summary, above every region boundary.
  landPayloads(harness.window, { ORCHESTRATORS: {} });

  let root;
  assert.doesNotThrow(() => {
    root = harness.mountPage();
  });

  const page = { ...harness, root };
  const { pageRenderFailureNotice } = page.window.DF_RENDER_FAILURE;
  const detail = assertOneNotice(page, 'page-render-failure-banner', pageRenderFailureNotice(undefined).text);
  assert.equal(failureNotices(root)[0].getAttribute('role'), 'alert');
  assert.ok(detail.startsWith('TypeError'), detail);
  assert.equal(root.querySelector('.app'), null, 'the page that threw must not render beside its notice');
  assertReportsOnlyContainedThrows(page.consoleErrors);
});

// ── Decision 7: the sidebar's check-owner section (escalation_check_owner.js::checkOwnerPanel) ──
// A dependency_capability record names two tasks: its own task_id is the
// blocked dependent, and delivered_check_owner.task_id is the owner whose
// declaration sites are edited. The fixture ids differ, so a swap is caught.

const CHECK_OWNER_TASK = '7514';
const CHECK_DEPENDENT_TASK = '7600';
const CHECK_SIDECAR_PATH = 'plans/x.capabilities.yaml';

function checkOwnerRow(deliveredCheckOwner) {
  return escalationRow({
    id: `esc-${CHECK_DEPENDENT_TASK}-1`,
    taskId: CHECK_DEPENDENT_TASK,
    summary: 'delivered check graph-index-provisioned fails on main',
    project: PROJECT,
    deliveredCheckOwner,
  });
}

function declaredCheckOwner(sidecar) {
  return { task_id: CHECK_OWNER_TASK, check_name: 'graph-index-provisioned', sidecar };
}

function shownTaskId(window, id) {
  return `T-${window.DF_SHELL.taskId(id)}`;
}

// Land `row` as the only escalation, open the tab, click the row, and return
// the sidebar it opens.
function openSidebar(harness, row) {
  const { window } = harness;
  seedDashboardData(window, {
    ESCALATIONS: escalationsPayload([{ project: PROJECT, rows: [row] }]),
    __receipt: receiptsFor(window.DF_DATA_LOADER, ['ESCALATIONS']),
  });
  const { container } = harness.mount(window.DF_TABS.EscalationsTab, { projectFilter: [] });
  const tableRow = [...container.querySelectorAll('tbody tr')].find(
    tr => tr.querySelector('td')?.textContent === shownTaskId(window, row.task_id),
  );
  assert.ok(tableRow, "the escalation's row did not render, so its sidebar cannot be opened");
  click(window, tableRow);
  const dialog = container.querySelector('[role="dialog"]');
  assert.ok(dialog, 'clicking the row opened no sidebar');
  return dialog;
}

function oneByTestId(root, testId) {
  const found = root.querySelectorAll(`[data-testid="${testId}"]`);
  assert.equal(found.length, 1, `expected exactly one [data-testid="${testId}"], found ${found.length}`);
  return found[0];
}

test("tab_escalations.jsx::EscalationSidebar renders checkOwnerPanel's owner, check, declaration sites and dependent", t => {
  const harness = openWindow(t);
  const row = checkOwnerRow(declaredCheckOwner({ path: CHECK_SIDECAR_PATH, on_main: 'declares' }));
  const dialog = openSidebar(harness, row);

  const panel = harness.window.DF_ESCALATION_CHECK_OWNER.checkOwnerPanel(row);
  const owner = shownTaskId(harness.window, panel.ownerTaskId);
  const dependent = shownTaskId(harness.window, panel.dependentTaskId);
  assert.notEqual(owner, dependent, 'the fixture must make the owner and the dependent differ');
  const section = oneByTestId(dialog, 'esc-check-owner');

  const ownerLine = oneByTestId(section, 'esc-check-owner-owner').textContent;
  assert.ok(ownerLine.includes(owner) && ownerLine.includes(panel.checkName), ownerLine);
  assert.ok(!ownerLine.includes(dependent), `the owner line names the dependent: ${ownerLine}`);

  const metadataSite = oneByTestId(section, 'esc-check-owner-metadata').textContent;
  assert.ok(metadataSite.includes(owner) && metadataSite.includes(panel.checkName), metadataSite);

  const sidecarSite = oneByTestId(section, 'esc-check-owner-sidecar').textContent;
  assert.ok(sidecarSite.includes(panel.sidecarPath) && sidecarSite.includes(panel.sidecarCaption), sidecarSite);

  const dependentLine = oneByTestId(section, 'esc-check-owner-dependent').textContent;
  assert.ok(dependentLine.includes(dependent), dependentLine);
  assert.ok(!dependentLine.includes(owner), `the dependent line names the owner: ${dependentLine}`);
});

test("tab_escalations.jsx::EscalationSidebar tones an unverified sidecar's caption with checkOwnerPanel's tone", t => {
  const harness = openWindow(t);
  const row = checkOwnerRow(declaredCheckOwner({ path: CHECK_SIDECAR_PATH, on_main: 'unverified' }));
  const dialog = openSidebar(harness, row);

  const panel = harness.window.DF_ESCALATION_CHECK_OWNER.checkOwnerPanel(row);
  assert.ok(panel.sidecarTone, 'the fixture must be a toned sidecar state');
  const captions = [...oneByTestId(dialog, 'esc-check-owner-sidecar').querySelectorAll('span')].filter(
    span => span.textContent === panel.sidecarCaption,
  );
  assert.equal(captions.length, 1, 'the sidecar site must render its caption in exactly one span');
  assert.ok(captions[0].classList.contains(panel.sidecarTone), captions[0].className);
});

test("tab_escalations.jsx::EscalationSidebar renders checkOwnerPanel's no-sidecar caption and no sidecar path", t => {
  const harness = openWindow(t);
  const row = checkOwnerRow(declaredCheckOwner(null));
  const dialog = openSidebar(harness, row);

  const panel = harness.window.DF_ESCALATION_CHECK_OWNER.checkOwnerPanel(row);
  assert.equal(panel.sidecarPath, null);
  const sidecarSite = oneByTestId(dialog, 'esc-check-owner-sidecar');
  assert.ok(sidecarSite.textContent.includes(panel.sidecarCaption), sidecarSite.textContent);
  assert.equal(sidecarSite.querySelector('.mono'), null, 'a site with no sidecar renders no sidecar path');
  assert.ok(!dialog.textContent.includes(CHECK_SIDECAR_PATH), 'the sidebar names a sidecar path it was not given');
  oneByTestId(dialog, 'esc-check-owner-owner');
});

test("tab_escalations.jsx::EscalationSidebar renders a sidecar with no path or on_main as a sidecar, in checkOwnerPanel's words", t => {
  const harness = openWindow(t);
  const row = checkOwnerRow(declaredCheckOwner({}));
  const dialog = openSidebar(harness, row);

  const panel = harness.window.DF_ESCALATION_CHECK_OWNER.checkOwnerPanel(row);
  const sidecarSite = oneByTestId(dialog, 'esc-check-owner-sidecar');
  assert.equal(sidecarSite.querySelector('.mono')?.textContent, panel.sidecarPath, 'a pathless sidecar must keep its path slot');
  assert.ok(sidecarSite.textContent.includes(panel.sidecarCaption), sidecarSite.textContent);
  assert.ok(!dialog.textContent.includes('undefined'), dialog.textContent);
});

test('tab_escalations.jsx::EscalationSidebar renders no check-owner section without an owner, the same for an absent key as for null', t => {
  const absentHarness = openWindow(t);
  const absent = checkOwnerRow(undefined);
  assert.equal('delivered_check_owner' in absent, false, 'the fixture must make absence real, not undefined-valued');
  const absentDialog = openSidebar(absentHarness, absent);
  assert.equal(absentHarness.window.DF_ESCALATION_CHECK_OWNER.checkOwnerPanel(absent), null);
  assert.equal(absentDialog.querySelector('[data-testid="esc-check-owner"]'), null);

  const nullHarness = openWindow(t);
  const nullDialog = openSidebar(nullHarness, checkOwnerRow(null));

  assert.equal(nullDialog.innerHTML, absentDialog.innerHTML);
});

// ── The sidebar's linked-task description (data.js::ON_DEMAND_KEYS.taskProse) ──
// Task rows carry no prose, so the drawer reads the linked task's description
// through the on-demand read the Task Detail pane uses, addressed by the uid
// active_tasks.task_uid builds: `<project>/T-<id>`.

const PROSE_TASK = '7700';
const PROSE_URL_PREFIX = '/api/v2/dashboard/task/';

function linkedTaskRow(task) {
  return escalationRow({ id: `esc-${PROSE_TASK}-1`, taskId: PROSE_TASK, summary: 'needs a ruling', project: PROJECT, task });
}

// Answers the taskProse url for `uid` with `response`, a fetch Response or a
// promise of one. Returns every url asked for.
function stubProseFetch(window, uid, response) {
  return stubFetch(window, { [window.DF_DATA_LOADER.ON_DEMAND_KEYS.taskProse.url(uid)]: response });
}

function proseResponse(window, uid, prose) {
  const { key } = window.DF_DATA_LOADER.ON_DEMAND_KEYS.taskProse;
  return { ok: true, status: 200, json: async () => ({ [key(uid)]: prose }) };
}

const PROSE_UNAVAILABLE = 'The description of this task could not be fetched.';

test("tab_escalations.jsx::EscalationSidebar renders the linked task's description from the taskProse on-demand read", async t => {
  const harness = openWindow(t);
  const { window } = harness;
  allowAsyncAct(window);
  const uid = `${PROJECT}/T-${PROSE_TASK}`;
  const asked = stubProseFetch(window, uid, proseResponse(window, uid, { description: 'why it exists', details: '' }));
  const row = linkedTaskRow(undefined);
  assert.equal('description' in row.task.value, false, 'the card itself must carry no description');

  const dialog = openSidebar(harness, row);
  await settle(window);

  assert.ok(dialog.textContent.includes('why it exists'), dialog.textContent);
  assert.deepEqual(
    asked.filter(url => url.startsWith(PROSE_URL_PREFIX)),
    [window.DF_DATA_LOADER.ON_DEMAND_KEYS.taskProse.url(uid)],
  );
});

test('tab_escalations.jsx::EscalationSidebar requests no taskProse for a card that is a hole', async t => {
  const harness = openWindow(t);
  const { window } = harness;
  allowAsyncAct(window);
  const uid = `${PROJECT}/T-${PROSE_TASK}`;
  const asked = stubProseFetch(window, uid, proseResponse(window, uid, { description: 'why it exists', details: '' }));
  const row = linkedTaskRow(wireDatum(null, { state: 'unknown', reason: 'the task could not be read' }));

  const dialog = openSidebar(harness, row);
  await settle(window);

  assert.ok(dialog.textContent.includes('the task could not be read'), dialog.textContent);
  assert.deepEqual(asked.filter(url => url.startsWith(PROSE_URL_PREFIX)), []);
});

test("tab_escalations.jsx::EscalationSidebar shows loading… until the linked task's taskProse read answers", async t => {
  const harness = openWindow(t);
  const { window } = harness;
  allowAsyncAct(window);
  const uid = `${PROJECT}/T-${PROSE_TASK}`;
  let answer;
  stubProseFetch(window, uid, new Promise(resolve => { answer = resolve; }));

  const dialog = openSidebar(harness, linkedTaskRow(undefined));
  await settle(window);
  assert.ok(dialog.textContent.includes('loading…'), dialog.textContent);

  answer(proseResponse(window, uid, { description: 'why it exists', details: '' }));
  await settle(window);
  assert.ok(dialog.textContent.includes('why it exists'), dialog.textContent);
  assert.equal(dialog.textContent.includes('loading…'), false, dialog.textContent);
});

test("tab_escalations.jsx::EscalationSidebar says the linked task's description could not be fetched when its taskProse read fails", async t => {
  const harness = openWindow(t);
  const { window } = harness;
  allowAsyncAct(window);
  const uid = `${PROJECT}/T-${PROSE_TASK}`;
  stubProseFetch(window, uid, { ok: false, status: 503, json: async () => ({}) });

  const dialog = openSidebar(harness, linkedTaskRow(undefined));
  await settle(window);

  assert.ok(dialog.textContent.includes(PROSE_UNAVAILABLE), dialog.textContent);
  assert.equal(dialog.textContent.includes('loading…'), false, dialog.textContent);
});

// ── Decision 8: the Tasks tab's preferences go through persisted_state.js ──
// A key exists only while its value differs from its default, and a write the
// browser refuses is reported on console.warn rather than swallowed.

const TASKS_PREFERENCE_KEYS = ['df.tasksOpen', 'df.tasksGroupByPrd', 'df.tasks.views'];

function storedKeys(window) {
  const { localStorage } = window;
  return Array.from({ length: localStorage.length }, (_, i) => localStorage.key(i));
}

function viewButton(window, container, key) {
  const { label } = window.DF_TASK_SNAPSHOT.CENSUS_VIEWS.find(view => view.key === key);
  const buttons = [...container.querySelectorAll('.seg button')].filter(button => button.textContent === label);
  assert.equal(buttons.length, 1, `expected one '${label}' view button`);
  return buttons[0];
}

function mountTasks(harness) {
  const task = inProgressTask(false);
  seedOneTask(harness.window, task);
  listFlat(harness.window);
  const handle = harness.mount(harness.window.DF_TASKS.TasksTab, TAB_PROPS);
  assert.ok(graphNode(handle.container, task), 'the seeded task never reached the graph');
  return handle;
}

function projectHead(container) {
  const heads = [...container.querySelectorAll('.proj-head')].filter(head => head.querySelector('.name')?.textContent === PROJECT);
  assert.equal(heads.length, 1, `expected one header for project ${PROJECT}`);
  return heads[0];
}

// Fills storage until jsdom's own quota refuses a single character more, and
// returns the error that refusal threw.
function fillStorageToQuota(storage) {
  let refusal = null;
  let written = 0;
  for (let size = 1 << 20; size >= 1; size >>= 1) {
    const chunk = 'x'.repeat(size);
    for (;;) {
      try {
        storage.setItem(`fill.${written}`, chunk);
        written += 1;
      } catch (error) {
        refusal = error;
        break;
      }
    }
  }
  return refusal;
}

test('tab_tasks.jsx::TasksTab stores none of its preferences while each is at its default', t => {
  const harness = openWindow(t);
  seedOneTask(harness.window, inProgressTask(false));
  harness.mount(harness.window.DF_TASKS.TasksTab, TAB_PROPS);

  const stored = storedKeys(harness.window);
  assert.deepEqual(TASKS_PREFERENCE_KEYS.filter(key => stored.includes(key)), []);
});

test('tab_tasks.jsx::TasksTab reports a preference it cannot store on console.warn, and the view still flips', t => {
  const harness = openWindow(t);
  const { window } = harness;
  listFlat(window);
  const refusal = fillStorageToQuota(window.localStorage);
  assert.equal(refusal?.name, 'QuotaExceededError', 'the fill did not reach the storage quota');
  const warnings = [];
  window.console.warn = (...args) => warnings.push(args);

  const { container } = mountTasks(harness);
  click(window, viewButton(window, container, 'terminal'));

  assert.ok(viewButton(window, container, 'terminal').classList.contains('on'), 'the terminal view did not flip on');
  assert.ok(
    warnings.some(([message, error]) => String(message).includes('df.tasks.views') && error?.name === 'QuotaExceededError'),
    `no console.warn named 'df.tasks.views' with its QuotaExceededError: ${JSON.stringify(warnings.map(([message]) => String(message)))}`,
  );
});

test('tab_tasks.jsx::TasksTab stores a flipped view, seeds the next mount with it, and removes the key when flipped back', t => {
  const harness = openWindow(t);
  const { window } = harness;
  const first = mountTasks(harness);
  click(window, viewButton(window, first.container, 'terminal'));
  assert.equal(JSON.parse(window.localStorage.getItem('df.tasks.views'))?.terminal, true, 'the flipped view was not stored');
  first.unmount();

  const { container } = mountTasks(harness);
  assert.ok(viewButton(window, container, 'terminal').classList.contains('on'), 'the stored view did not seed the next mount');
  click(window, viewButton(window, container, 'terminal'));
  assert.ok(!storedKeys(window).includes('df.tasks.views'), 'the view is back at its default, yet its key is still stored');
});

test('tab_tasks.jsx::TasksTab folds a project on its first header click, and the fold seeds the next mount', t => {
  const harness = openWindow(t);
  const { window } = harness;
  const first = mountTasks(harness);
  assert.equal(projectHead(first.container).dataset.open, 'true', 'a project with no stored flag did not start open');
  click(window, projectHead(first.container));
  assert.equal(projectHead(first.container).dataset.open, 'false', 'one header click did not fold the project');
  assert.ok(storedKeys(window).includes('df.tasksOpen'), 'the fold was not stored');
  first.unmount();

  // A folded project renders no graph, so this mount cannot go through mountTasks.
  const { container } = harness.mount(window.DF_TASKS.TasksTab, TAB_PROPS);
  assert.equal(projectHead(container).dataset.open, 'false', 'the stored fold did not seed the next mount');
  click(window, projectHead(container));
  assert.equal(projectHead(container).dataset.open, 'true', 'one header click did not unfold the project');
  assert.ok(!storedKeys(window).includes('df.tasksOpen'), 'the project is back at its default-open, yet its key is still stored');
});

function groupingButton(container, label) {
  const buttons = [...projectHead(container).querySelectorAll('.seg button')].filter(button => button.textContent === label);
  assert.equal(buttons.length, 1, `expected one '${label}' grouping button`);
  return buttons[0];
}

test('tab_tasks.jsx::TasksTab stores a project listed flat, and removes the key when grouped by PRD again', t => {
  const harness = openWindow(t);
  const { window } = harness;
  seedOneTask(window, inProgressTask(false));
  const { container } = harness.mount(window.DF_TASKS.TasksTab, TAB_PROPS);
  assert.ok(groupingButton(container, 'by PRD').classList.contains('on'), 'by PRD is not the default grouping');
  click(window, groupingButton(container, 'list'));
  assert.equal(JSON.parse(window.localStorage.getItem('df.tasksGroupByPrd'))?.[PROJECT], 'flat', 'the flat listing was not stored');

  click(window, groupingButton(container, 'by PRD'));
  assert.ok(groupingButton(container, 'by PRD').classList.contains('on'), 'the project did not return to the lanes');
  assert.ok(!storedKeys(window).includes('df.tasksGroupByPrd'), 'the grouping is back at its default, yet its key is still stored');
});

// ── Decision 9: a count axis's maximum (spark_path.js::axisMax) ──
// A chart marked `countAxis` labels its gridlines and draws its data against
// axisMax's snapped maximum; any other chart keeps its exact data maximum, and
// formatY only presents the ticks it is handed.

const DATA_MAX = 7;

const COUNT_AXIS_CHARTS = [
  {
    name: 'LineChart',
    props: { series: [{ values: [0, DATA_MAX] }], labels: ['a', 'b'] },
  },
  {
    name: 'StackedAreaChart',
    props: { stacks: [{ key: 'a', color: 'red', values: [3, DATA_MAX] }], labels: ['a', 'b'] },
  },
];

function yTickLabels(container) {
  return [...container.querySelectorAll('svg text[text-anchor="end"]')].map(text => text.textContent);
}

function gridlineYs(container) {
  return [...container.querySelectorAll('svg line')].map(line => Number(line.getAttribute('y1')));
}

function peakY(container) {
  const ys = [...container.querySelectorAll('svg path')].flatMap(path =>
    [...path.getAttribute('d').matchAll(/[ML]\s*[-\d.e]+,([-\d.e]+)/g)].map(match => Number(match[1])),
  );
  return Math.min(...ys);
}

function renderAxis(harness, chart, props) {
  const { container } = harness.mount(harness.window.DF_CHARTS[chart.name], { ...chart.props, ...props });
  const labels = yTickLabels(container);
  const gridYs = gridlineYs(container);
  return {
    labels,
    ticks: labels.length - 1,
    floor: Math.max(...gridYs),
    top: Math.min(...gridYs),
    peak: peakY(container),
  };
}

function tickValues(max, ticks) {
  return Array.from({ length: ticks + 1 }, (_, i) => (max * i) / ticks);
}

for (const chart of COUNT_AXIS_CHARTS) {
  test(`charts.jsx::${chart.name} with countAxis labels its gridlines to axisMax's snapped maximum`, t => {
    const harness = openWindow(t);
    const axis = renderAxis(harness, chart, { countAxis: true });
    const max = harness.window.DF_SPARK_PATH.axisMax(DATA_MAX, axis.ticks, true);
    assert.notEqual(max, DATA_MAX, 'the fixture does not make the snap visible');
    assert.deepEqual(axis.labels, tickValues(max, axis.ticks).map(String));
  });

  test(`charts.jsx::${chart.name} with countAxis draws its data on the same snapped scale as its gridlines`, t => {
    const harness = openWindow(t);
    const axis = renderAxis(harness, chart, { countAxis: true });
    const max = harness.window.DF_SPARK_PATH.axisMax(DATA_MAX, axis.ticks, true);
    assert.equal((axis.floor - axis.peak) / (axis.floor - axis.top), DATA_MAX / max);
  });

  test(`charts.jsx::${chart.name} without countAxis keeps its exact data maximum`, t => {
    const harness = openWindow(t);
    const axis = renderAxis(harness, chart, { formatY: v => String(v) });
    const max = harness.window.DF_SPARK_PATH.axisMax(DATA_MAX, axis.ticks, false);
    assert.deepEqual(axis.labels, tickValues(max, axis.ticks).map(String));
    assert.equal(axis.peak, axis.top, 'the data maximum does not reach the top gridline');
  });

  test(`charts.jsx::${chart.name} hands a caller's formatY the snapped ticks on a count axis`, t => {
    const harness = openWindow(t);
    const axis = renderAxis(harness, chart, { countAxis: true, formatY: v => `${v} ops` });
    const max = harness.window.DF_SPARK_PATH.axisMax(DATA_MAX, axis.ticks, true);
    assert.deepEqual(axis.labels, tickValues(max, axis.ticks).map(v => `${v} ops`));
  });
}

// ── Decision 10: the Tasks tab's banner notices (tasks_offline_banner.js::tasksBannerNoticesFor) ──
// Every notice kind the helper returns renders under its own test id. Each
// case seeds the one /tasks fact that yields that kind and nothing else, so a
// kind with no bannerTestIds entry renders with no test id and fails here.

const OTHER_PROJECT = 'q';
const BANNER_CASES = [
  { kind: 'global', testId: 'tasks-offline-banner',
    data: { TASKS_OFFLINE: true, TASKS_OFFLINE_PROJECTS: [PROJECT], TASKS_PROJECT_COUNT: 1 } },
  { kind: 'partial', testId: 'tasks-partial-banner',
    data: { TASKS_OFFLINE_PROJECTS: [OTHER_PROJECT], TASKS_PROJECT_COUNT: 2 } },
  { kind: 'degraded', testId: 'tasks-degraded-banner',
    data: { TASKS_DEGRADED_PROJECTS: [OTHER_PROJECT], TASKS_PROJECT_COUNT: 2 } },
  { kind: 'count-unknown', testId: 'tasks-count-unknown-banner',
    data: { TASKS_COUNT_UNKNOWN_PROJECTS: [PROJECT], TASKS_PROJECT_COUNT: 1 } },
  { kind: 'count-stale', testId: 'tasks-count-stale-banner',
    data: { TASKS_COUNT_STALE_PROJECTS: [PROJECT], TASKS_PROJECT_COUNT: 1 } },
];

for (const { kind, testId, data } of BANNER_CASES) {
  test(`tab_tasks.jsx::TasksTab renders the ${kind} notice tasksBannerNoticesFor describes under ${testId}`, t => {
    const harness = openWindow(t);
    const { window } = harness;
    seedOneTask(window, inProgressTask(false));
    seedDashboardData(window, { ...data, __receipt: receiptsFor(window.DF_DATA_LOADER, Object.keys(data)) });
    const { container } = harness.mount(window.DF_TASKS.TasksTab, TAB_PROPS);

    const notices = window.DF_TASKS_OFFLINE_BANNER.tasksBannerNoticesFor(window.DF_DATA);
    assert.deepEqual(Array.from(notices, n => n.kind), [kind]);
    const rendered = [...container.querySelectorAll(`[data-testid="${testId}"]`)];
    assert.equal(rendered.length, 1, `the ${kind} notice must render exactly once under ${testId}`);
    assert.equal(rendered[0].textContent, notices[0].text);
    for (const other of BANNER_CASES.filter(c => c.kind !== kind)) {
      assert.deepEqual([...container.querySelectorAll(`[data-testid="${other.testId}"]`)], [], other.testId);
    }
  });
}

// ── Decision 11: the Task-detail Module locks section (task_row_cells.js::locksCellState, scheduler_utils.jsx::lockChipStateFor) ──

function seedTaskWithScheduler(window, task, scheduler) {
  seedOneTask(window, task);
  seedDashboardData(window, { SCHEDULER: scheduler, __receipt: receiptsFor(window.DF_DATA_LOADER, ['SCHEDULER']) });
}

// The "Module locks" label and the element it heads, or null when the pane
// draws no such section.
function moduleLocksSection(detail) {
  const label = [...detail.querySelectorAll('.section-lbl')].find(lbl => lbl.textContent.startsWith('Module locks'));
  return label ? { label, body: label.nextElementSibling } : null;
}

// What the section is asked to draw, as the helpers it calls answer it.
function expectedLocksCell(window, task) {
  const D = window.DF_DATA;
  const { locksCellState, schedulerLocksDatum } = window.DF_TASK_ROW_CELLS;
  return locksCellState(schedulerLocksDatum(D.SCHEDULER, PROJECT, D.__receipt), window.DF_SCHED_UTILS.buildSchedLockInfo(task, D.SCHEDULER));
}

function lockChips(root) {
  return [...root.querySelectorAll('.chip')].filter(chip => [...chip.classList].some(cls => cls.startsWith('lock-')));
}

test("tab_tasks.jsx::TaskDetail draws the Module locks em-dash with locksCellState's reason for a project the scheduler lists offline", t => {
  const harness = openWindow(t);
  const task = inProgressTask(false);
  seedTaskWithScheduler(harness.window, task, schedulerSnapshot({ offlineProjects: [PROJECT] }));
  const detail = taskDetailOf(harness, task);
  assert.ok(detail, 'the detail pane never opened on the task');

  const expected = expectedLocksCell(harness.window, task);
  assert.ok(expected.placeholder, 'locksCellState draws no placeholder for an offline project, so this fixture tests nothing');
  assert.ok(expected.title, 'locksCellState gives no reason for an offline project, so this fixture tests nothing');
  const section = moduleLocksSection(detail);
  assert.ok(section, 'the detail pane draws no Module locks section for a project whose locks are unknown');
  assert.equal(section.label.textContent, 'Module locks', 'an unknown lock set has no count to show');
  assert.ok(section.body?.matches('span.chip-empty'), 'the unknown arm must draw a span.chip-empty under the label');
  assert.equal(section.body.textContent, expected.placeholder);
  assert.equal(section.body.title, expected.title, "the em-dash must carry locksCellState's reason as its title");
  assert.deepEqual(lockChips(detail), [], 'a project whose locks are unknown must draw no lock chip');
});

// `task` locks four paths in a measured project, one in each lock class.
function seedTaskHoldingEveryLockClass(window, task) {
  const paths = { mine: 'src/mine', taken: 'src/taken', parked: 'src/parked', free: 'src/free' };
  seedTaskWithScheduler(window, task, schedulerSnapshot({
    rows: [schedulerLockRow({ project: PROJECT, taskId: task.id, lockSet: Object.values(paths) })],
    modules: [
      schedulerModule({ project: PROJECT, path: paths.mine, holder: task.id, holderProject: PROJECT }),
      schedulerModule({ project: PROJECT, path: paths.taken, holder: '7', holderProject: PROJECT }),
      schedulerModule({ project: PROJECT, path: paths.parked, parkedBy: '9', parkedOwnerLive: true }),
    ],
  }));
}

test('tab_tasks.jsx::TaskDetail draws each lock chip in the class lockChipStateFor answers, for a project the scheduler measured', t => {
  const harness = openWindow(t);
  const { window } = harness;
  const task = inProgressTask(false);
  seedTaskHoldingEveryLockClass(window, task);
  const detail = taskDetailOf(harness, task);
  assert.ok(detail, 'the detail pane never opened on the task');

  const lockInfo = window.DF_SCHED_UTILS.buildSchedLockInfo(task, window.DF_DATA.SCHEDULER);
  assert.equal(expectedLocksCell(window, task).placeholder, null, 'locksCellState reads this measured project as unknown');
  const expected = new Map(lockInfo.lockSet.map(path => [
    path,
    window.DF_SCHED_UTILS.lockChipStateFor(lockInfo.moduleByPath.get(path), task.id, PROJECT),
  ]));
  assert.equal(new Set([...expected.values()].map(state => state.cls)).size, 4, 'the four paths must cover four lock classes');

  const section = moduleLocksSection(detail);
  assert.ok(section, 'the detail pane draws no Module locks section for a task holding locks');
  assert.equal(section.label.textContent, `Module locks (${lockInfo.lockSet.length})`);
  assert.ok(section.body && !section.body.matches('.chip-empty') && !section.body.querySelector('.chip-empty'),
    'a measured project with locks must draw chips, not the em-dash');
  for (const [path, { cls, ownerLabel }] of expected) {
    const chip = section.body.querySelector(`.chip[title="${path}"]`);
    assert.ok(chip, `no chip titled ${path}`);
    assert.ok(chip.classList.contains(cls), `the ${path} chip is not ${cls}, the class lockChipStateFor answers`);
    if (cls === 'lock-taken' || cls === 'lock-parked') {
      assert.ok(chip.querySelector('.holder')?.textContent.includes(ownerLabel), `the ${path} chip does not name ${ownerLabel}`);
    }
  }
});

// An inline classifier answers as lockChipStateFor does until the two drift, so
// the public seam is swapped for a sentinel: a chip it did not classify shows.
test('tab_tasks.jsx::TaskDetail classifies every lock chip through window.DF_SCHED_UTILS.lockChipStateFor, with no classifier of its own', t => {
  const harness = openWindow(t);
  const { window } = harness;
  const task = inProgressTask(false);
  seedTaskHoldingEveryLockClass(window, task);
  const { lockSet } = window.DF_SCHED_UTILS.buildSchedLockInfo(task, window.DF_DATA.SCHEDULER);
  window.DF_SCHED_UTILS = { ...window.DF_SCHED_UTILS, lockChipStateFor: () => ({ cls: 'lock-probe', ownerLabel: 'probe' }) };
  const detail = taskDetailOf(harness, task);
  assert.ok(detail, 'the detail pane never opened on the task');

  const section = moduleLocksSection(detail);
  assert.ok(section, 'the detail pane draws no Module locks section for a task holding locks');
  const chips = lockChips(section.body);
  assert.equal(chips.length, lockSet.length, 'the section must draw one chip per locked path');
  for (const chip of chips) {
    assert.deepEqual([...chip.classList].filter(cls => cls.startsWith('lock-')), ['lock-probe'],
      `the ${chip.title} chip was not classified by window.DF_SCHED_UTILS.lockChipStateFor`);
  }
});

test('tab_tasks.jsx::TaskDetail draws no Module locks section for a task holding no locks in a project the scheduler measured', t => {
  const harness = openWindow(t);
  const task = inProgressTask(false);
  seedTaskWithScheduler(harness.window, task, schedulerSnapshot());
  const detail = taskDetailOf(harness, task);
  assert.ok(detail, 'the detail pane never opened on the task, so the absent section proves nothing');

  assert.equal(expectedLocksCell(harness.window, task).placeholder, null, 'locksCellState reads this measured project as unknown');
  assert.equal(moduleLocksSection(detail), null, 'a measured project with no locks draws no Module locks section');
});

// ── The memory-eval section (tab_memory_evals.jsx::MemoryEvalsSection) ──

const MEMORY_EVAL_ID = 'e1-retrieval-health';
const TRIPWIRE = 'topic-canonical-present';

// The section mounted over `payload`, with every onNavigate call recorded.
function mountMemoryEvals(t, payload) {
  const harness = openWindow(t);
  const { window } = harness;
  seedDashboardData(window, {
    MEMORY_EVALS: payload,
    __receipt: receiptsFor(window.DF_DATA_LOADER, ['MEMORY_EVALS']),
  });
  const navigations = [];
  const { container } = harness.mount(window.DF_MEMORY_EVALS.MemoryEvalsSection, {
    onNavigate: (...args) => navigations.push(args),
  });
  return { window, container, navigations };
}

// The rendered table rows, one per metric row, in payload order.
function metricRowElements(container, rows) {
  const trs = [...container.querySelectorAll('tbody tr')];
  assert.equal(trs.length, rows.length, 'each metric row must render as exactly one table row');
  return trs;
}

function escalationLinks(root) {
  return [...root.querySelectorAll('[data-testid="memory-eval-escalation-link"]')];
}

// The one span in `tr` that renders verdictBadge(row): its words, its classes,
// and the evaluator's verdict_detail as its title.
function renderedBadge(window, tr, row) {
  const badge = window.DF_MEMORY_EVALS_FMT.verdictBadge(row);
  const spans = spansDescribedBy(tr, { ...badge, title: row.verdict_detail || '' });
  assert.equal(spans.length, 1, `${row.metric_id} ${row.item ?? ''}: expected one span reading "${badge.label}"`);
  return spans[0];
}

function assertLinksOnly(tr, row) {
  const links = escalationLinks(tr);
  if (!row.escalation) {
    assert.equal(links.length, 0, `${row.metric_id} ${row.item ?? ''} (${row.parity}) renders an escalation link`);
    return;
  }
  assert.equal(links.length, 1, `${row.metric_id} ${row.item ?? ''} (${row.parity}) must render one escalation link`);
  assert.ok(links[0].textContent.includes(row.escalation.id), links[0].textContent);
}

test("tab_memory_evals.jsx::MemoryEvalsSection renders each parity state in verdictBadge's words and links only open escalations", t => {
  const alarmed = memoryEvalEscalation('esc-alarmed');
  const recovered = memoryEvalEscalation('esc-recovered');
  const rows = [
    memoryEvalRow({
      verdict: 'alarm', verdict_detail: '12/30 against a pooled baseline of 72/90 over 3 run(s).',
      fingerprint: alarmed.dedupe_fingerprint, escalation: alarmed, parity: 'alarmed_open',
    }),
    memoryEvalRow({
      metric_id: 'dangling-pointers', kind: 'count', rule_kind: 'count',
      verdict: 'no_alarm', fingerprint: recovered.dedupe_fingerprint, escalation: recovered, parity: 'recovered_open',
    }),
    memoryEvalRow({
      metric_id: 'superseded-above-successor', kind: 'count', rule_kind: 'count',
      verdict: 'insufficient_data', fingerprint: 'fp-insufficient', parity: 'insufficient_data',
    }),
    memoryEvalRow({
      metric_id: TRIPWIRE, kind: 'tripwire', rule_kind: 'tripwire', item: 't-recon-watcher-triage',
      verdict: 'grandfathered', fingerprint: 'fp-grandfathered', parity: 'grandfathered',
    }),
  ];
  const payload = memoryEvalsPayload({ evals: [memoryEvalCard(MEMORY_EVAL_ID, rows)] });
  const { window, container, navigations } = mountMemoryEvals(t, payload);

  const trs = metricRowElements(container, rows);
  rows.forEach((row, i) => {
    const badge = renderedBadge(window, trs[i], row);
    assertLinksOnly(trs[i], row);
    if (!row.escalation) assert.ok(!badge.classList.contains('bad'), `${row.parity} renders alarm-shaped: ${badge.className}`);
  });

  click(window, escalationLinks(trs[0])[0]);
  assert.equal(navigations.length, 1);
  const [view, target] = navigations[0];
  assert.equal(view, 'esc');
  assert.deepEqual({ ...target }, { queue: payload.escalation_queue, id: alarmed.id });
});

test('tab_memory_evals.jsx::MemoryEvalsSection under a storm renders one aggregate banner and no per-metric links', t => {
  const aggregate = memoryEvalEscalation('esc-storm-aggregate');
  const storm = {
    triggered: true, alarm_count: 3, aggregate_fingerprint: aggregate.dedupe_fingerprint, escalation: aggregate,
  };
  const collapsed = ['canonical-in-top-5', 'dangling-pointers', 'superseded-above-successor'].map(metricId =>
    memoryEvalRow({ metric_id: metricId, verdict: 'alarm', fingerprint: `fp-${metricId}`, parity: 'storm_collapsed' }),
  );
  const grandfathered = memoryEvalRow({
    metric_id: TRIPWIRE, kind: 'tripwire', rule_kind: 'tripwire', item: 't-recon-watcher-triage',
    verdict: 'grandfathered', fingerprint: 'fp-grandfathered', parity: 'grandfathered',
  });
  const rows = [...collapsed, grandfathered];
  const payload = memoryEvalsPayload({ evals: [memoryEvalCard(MEMORY_EVAL_ID, rows)], storm_escape: storm });
  const { window, container } = mountMemoryEvals(t, payload);

  const banner = oneByTestId(container, 'memory-eval-storm-banner');
  assert.ok(banner.textContent.includes(`${storm.alarm_count} alarms`), banner.textContent);
  const bannerLinks = escalationLinks(banner);
  assert.equal(bannerLinks.length, 1, 'the banner must link the aggregate escalation');
  assert.ok(bannerLinks[0].textContent.includes(aggregate.id), bannerLinks[0].textContent);

  const trs = metricRowElements(container, rows);
  rows.forEach((row, i) => {
    renderedBadge(window, trs[i], row);
    assertLinksOnly(trs[i], row);
    assert.equal(
      trs[i].textContent.includes('link suppressed — storm aggregate'),
      row.parity === 'storm_collapsed',
      `${row.metric_id} (${row.parity}) and the storm-suppression note disagree`,
    );
  });
  assert.equal(escalationLinks(container).length, 1, 'only the banner may link an escalation under a storm');
});

test("tab_memory_evals.jsx::MemoryEvalsSection renders a tripwire's per-item verdicts as one row per item, each naming its item", t => {
  const alarmed = memoryEvalEscalation('esc-item-1');
  const rows = [
    memoryEvalRow({ verdict: 'no_alarm', fingerprint: 'fp-window', parity: 'clear' }),
    memoryEvalRow({
      metric_id: TRIPWIRE, kind: 'tripwire', rule_kind: 'tripwire', item: 't-alpha',
      verdict: 'alarm', fingerprint: alarmed.dedupe_fingerprint, escalation: alarmed, parity: 'alarmed_open',
    }),
    memoryEvalRow({
      metric_id: TRIPWIRE, kind: 'tripwire', rule_kind: 'tripwire', item: 't-beta',
      verdict: 'grandfathered', fingerprint: 'fp-grandfathered', parity: 'grandfathered',
    }),
  ];
  const payload = memoryEvalsPayload({ evals: [memoryEvalCard(MEMORY_EVAL_ID, rows)] });
  const { window, container } = mountMemoryEvals(t, payload);

  const trs = metricRowElements(container, rows);
  const items = rows.map(row => row.item).filter(Boolean);
  rows.forEach((row, i) => {
    const firstCell = trs[i].querySelector('td').textContent;
    assert.ok(firstCell.includes(row.metric_id), firstCell);
    for (const item of items) {
      assert.equal(trs[i].textContent.includes(item), item === row.item, `the ${row.item ?? 'item-less'} row and ${item} disagree`);
    }
    if (row.item) assert.ok(firstCell.includes(row.item), `the item is not named beside its metric: ${firstCell}`);
    renderedBadge(window, trs[i], row);
    assertLinksOnly(trs[i], row);
  });
});

// ── Decision 12: the filed cell (task_row_cells.js::filedCellState) ──

// TaskDetail's 'filed' kv row: its key span and the value span beside it.
function filedRowOf(detail) {
  const key = [...detail.querySelectorAll('.kv > span.k')].find(span => span.textContent === 'filed');
  return { key, value: key?.nextElementSibling };
}

const FILED_CASES = [
  {
    name: 'a recorded filing time',
    task: () => ({ ...inProgressTask(false), created_at: '2026-11-12T09:10:17.253Z', created_at_source: 'first_event' }),
    recorded: true,
  },
  {
    name: 'no recorded filing time',
    task: () => ({ ...inProgressTask(false), created_at: null, created_at_source: null }),
    recorded: false,
  },
];

for (const { name, task: buildTask, recorded } of FILED_CASES) {
  test(`tab_tasks.jsx::TaskDetail renders the filed row filedCellState describes, for ${name}`, t => {
    const harness = openWindow(t);
    const task = buildTask();
    seedOneTask(harness.window, task);
    const detail = taskDetailOf(harness, task);
    assert.ok(detail, 'the detail pane never opened on the task');

    const expected = harness.window.DF_TASK_ROW_CELLS.filedCellState(task);
    const { key, value } = filedRowOf(detail);
    assert.ok(key, "TaskDetail renders no 'filed' kv key");
    assert.equal(value?.textContent, expected.text);
    assert.equal(value?.title, expected.title);
    const statusKey = key.previousElementSibling?.previousElementSibling;
    assert.ok(statusKey?.matches('span.k'), "the 'filed' row does not follow a kv row");
    assert.equal(statusKey.textContent, 'status', "the 'filed' row is not directly after the status row");

    if (recorded) {
      assert.notEqual(expected.text, harness.window.DF_DATUM.EM_DASH, 'fixture: the filing time is recorded');
      assert.ok(
        !detail.querySelector('.sub').textContent.includes(expected.text),
        "the filing time is rendered on the .sub line beside fmtAge's unlabelled reading",
      );
    }
  });
}
