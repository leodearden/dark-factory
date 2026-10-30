// The dashboard's JSX render seam: each render decision extracted into a
// classic-script helper must actually reach the DOM at every call site. The
// real .jsx renders in a jsdom window (dom_harness.mjs); every expected value
// comes from calling the helper itself, never from source text.
//
// All render tests live in this one file: node --test runs each file in its
// own process, and each process pays the Babel compile once.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { caughtFrom } from './_caught.mjs';
import { buildDashboardWindow, seedDashboardData } from './dom_harness.mjs';
import {
  burndownBlock,
  escalationAnalytics,
  escalationRow,
  escalationsPayload,
  mergeQueueBlock,
  openEscalationItem,
  orchestratorRow,
  projectRow,
  receiptsFor,
  taskCensus,
  tasksSnapshotEntry,
  wireDatum,
} from './wire_fixtures.mjs';

const PROJECT = 'p';
const TAB_PROPS = { projectFilter: [], search: '' };

// React reports a broken render (a missing key, an update outside act(), a bad
// prop) on console.error and renders on, so every test also asserts that
// nothing reached it, through the unmount.
function openWindow(t) {
  const harness = buildDashboardWindow();
  t.after(() => {
    harness.close();
    assert.deepEqual(harness.consoleErrors, [], 'the window wrote to console.error');
  });
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
    const caption = harness.window.DF_MERGE_QUEUE.depthBucketCaption(block.depth);
    assert.deepEqual(titles, [`Merge attempts · ${caption}`]);
  });
}

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
