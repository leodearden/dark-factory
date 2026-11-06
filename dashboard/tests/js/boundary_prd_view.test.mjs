// Boundary rows 20–21 of plans/dashboard-prd-lanes-prd.md: the PRD lanes page.
//
// Every body here is one the REAL /prds and /tasks routes served
// (test_boundary_js.py builds them through _boundary_payloads.prd_lanes_family
// and prd_view_family). The landing test lands /prds through data.js's real
// refreshOne; the render tests seed the same bodies into a dom_harness window
// and mount the real components, taking every expected text from
// prd_lanes_view.js's helpers and the served values.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { applyServed, loadClient, payload } from './_boundary_client.mjs';
import { buildDashboardWindow, seedDashboardData } from './dom_harness.mjs';
import { projectRow, receiptsFor, servingEndpoint } from './wire_fixtures.mjs';

const PRDS = '/api/v2/dashboard/prds';
const PROJECT = 'dark-factory';

test('the served /prds body lands through refreshOne and reads back per project, stamped', async () => {
  const client = loadClient();
  const body = payload('prd_lanes');
  const receivedAt = Date.parse(body.served_at);
  assert.equal(await applyServed(client, PRDS, body, { receivedAt }), 'applied');

  const win = client.window;
  const { projectPrdView, projectPrdContention } = win.DF_PRD_LANES_VIEW;
  const readings = [
    [projectPrdView(win.DF_DATA, PROJECT), body.PRD_VIEW[PROJECT]],
    [projectPrdContention(win.DF_DATA, PROJECT), body.PRD_CONTENTION[PROJECT]],
  ];
  for (const [read, served] of readings) {
    assert.deepEqual(read.value, served.value);
    assert.equal(read.state, served.state);
    assert.equal(read.as_of, served.as_of);
    assert.equal(read.reason, served.reason);
    assert.equal(read._received_at, receivedAt, 'the reader must stamp the /prds receipt');
  }

  const { TAB_ENDPOINTS } = win.DF_POLL_SCOPE;
  assert.ok(TAB_ENDPOINTS.tasks.includes(PRDS), 'the Tasks tab must poll /prds');
  const others = Object.entries(TAB_ENDPOINTS).filter(([tab, paths]) => tab !== 'tasks' && paths.includes(PRDS));
  assert.deepEqual(others.map(([tab]) => tab), [], 'no other tab polls /prds (§4.4)');
});

// ── Row 20: the lanes page, rendered from the served bodies ──

// React reports a broken render (a missing key, an update outside act()) on
// console.error and renders on, so every window asserts nothing reached it.
function servedWindow(t, prdsName, tasksName = null) {
  const harness = buildDashboardWindow();
  t.after(() => {
    harness.close();
    assert.deepEqual(harness.consoleErrors, [], 'the window wrote to console.error');
  });
  const { window } = harness;
  const loader = window.DF_DATA_LOADER;
  const patch = { __receipt: {} };
  const land = (body, keys) => {
    for (const key of keys) {
      patch[key] = body[key];
      patch.__receipt[servingEndpoint(loader, key).stateKey] = { servedAt: body.served_at, receivedAt: Date.now() };
    }
  };
  land(payload(prdsName), ['PRD_VIEW', 'PRD_CONTENTION']);
  if (tasksName !== null) land(payload(tasksName), ['TASKS_SNAPSHOT']);
  seedDashboardData(window, patch);
  return { harness, window, D: window.DF_DATA, helpers: window.DF_PRD_LANES_VIEW };
}

function click(window, element) {
  window.React.act(() => element.dispatchEvent(new window.MouseEvent('click', { bubbles: true })));
}

// ProjectPrdLanes over the window's served view (or `view`) and rows, with a
// member renderer that draws one li per row title, and every onSelect recorded.
function mountLanes(ctx, view = ctx.helpers.projectPrdView(ctx.D, PROJECT)) {
  const { window, D, helpers } = ctx;
  const { React } = window;
  const pool = window.DF_TASK_SNAPSHOT.projectRows(D, PROJECT).value || [];
  const selected = [];
  const renderMembers = rows => React.createElement(
    'ul', { className: 'members-stub' }, rows.map(row => React.createElement('li', { key: row.id }, row.title)),
  );
  const props = { project: PROJECT, view, pool, renderMembers, onSelect: id => selected.push(id) };
  const handle = ctx.harness.mount(window.DF_PRD_LANES.ProjectPrdLanes, props);
  return { root: handle.container, view, pool, selected, handle, props };
}

function rowsByKey(view) {
  return new Map(view.value.prds.map(row => [row.key, row]));
}

function expandTail(window, root) {
  click(window, root.querySelector('.prd-lane[data-lane="tail"] .prd-tail-line'));
}

// The title and the text of the one titled figure inside `scope`'s `selector`.
function reading(scope, selector) {
  const el = scope.querySelector(`${selector} [title]`);
  assert.ok(el, `no titled reading under ${selector}`);
  return { title: el.title, text: el.textContent };
}

// The one pip heading a block: the only reading of its Datum's provenance.
function headPip(scope, selector) {
  const pips = scope.querySelectorAll(`${selector} .pip`);
  assert.equal(pips.length, 1, `expected one pip heading ${selector}`);
  return { el: pips[0], title: pips[0].title, text: pips[0].textContent };
}

// Every figure a row head draws, as [cell, expected title, expected text].
function rowFigures(helpers, rules, row) {
  const { figureTitle, doneOfTotal, signalCell, readyCell } = helpers;
  const total = doneOfTotal(row);
  const signal = signalCell(row.signal);
  const figures = [
    ['.prd-title', figureTitle(rules.title, row.key), row.title],
    ['.prd-state', figureTitle(rules.state_line, null), row.state_line],
    ['.prd-total', figureTitle(rules.total, total.detail), total.text],
    ['.prd-signal', figureTitle(rules.signal, signal.detail), signal.text],
    ['.prd-ready', figureTitle(rules.ready, readyCell(row.ready).detail), readyCell(row.ready).text],
  ];
  if (row.next_action.kind !== 'none') figures.push(['.prd-action', figureTitle(rules.next_action, null), row.next_action.text]);
  return figures;
}

test('prd lanes #20: the lanes render in the served display order, each header titled with its rule', t => {
  const ctx = servedWindow(t, 'prd_lanes', 'prd_lanes_tasks');
  const { root, view } = mountLanes(ctx);
  const lanes = view.value.lanes;
  const sections = [...root.querySelectorAll('.prd-lane')];

  assert.equal(lanes.length, 5);
  assert.deepEqual(sections.map(section => section.dataset.lane), lanes.map(entry => entry.lane));
  sections.forEach((section, i) => {
    const head = reading(section, '.prd-lane-head');
    assert.equal(head.title, lanes[i].rule);
    assert.equal(head.text, `${lanes[i].label} (${lanes[i].prd_keys.length})`);
    if (lanes[i].lane === 'tail') return;
    assert.deepEqual([...section.querySelectorAll('.prd-row')].map(row => row.dataset.key), lanes[i].prd_keys);
  });
});

test('prd lanes #20: the tail lane renders collapsed to one line naming every tail PRD, and expands on click', t => {
  const ctx = servedWindow(t, 'prd_lanes', 'prd_lanes_tasks');
  const { root, view } = mountLanes(ctx);
  const byKey = rowsByKey(view);
  const entry = view.value.lanes.find(lane => lane.lane === 'tail');
  const tail = root.querySelector('.prd-lane[data-lane="tail"]');
  const line = tail.querySelectorAll('.prd-tail-line');

  assert.equal(line.length, 1, 'one collapsed line');
  assert.equal(tail.querySelectorAll('.prd-row').length, 0, 'a collapsed tail draws no row');
  assert.ok(line[0].textContent.startsWith(`${entry.prd_keys.length}: `), line[0].textContent);
  assert.ok(!line[0].textContent.includes('delivered'), `the tail line restates the tail rule: ${line[0].textContent}`);
  for (const key of entry.prd_keys) {
    assert.ok(line[0].textContent.includes(byKey.get(key).title), `${key}'s title is missing from the tail line`);
  }

  click(ctx.window, line[0]);
  assert.deepEqual([...tail.querySelectorAll('.prd-row')].map(row => row.dataset.key), entry.prd_keys);
});

test('prd lanes #20: every row bar draws the six served segments in legend order, under one legend titled with the segments rule', t => {
  const ctx = servedWindow(t, 'prd_lanes', 'prd_lanes_tasks');
  const { root, view } = mountLanes(ctx);
  const { barSegments } = ctx.helpers;
  const legend = view.value.segment_legend;
  const byKey = rowsByKey(view);
  expandTail(ctx.window, root);

  const legends = root.querySelectorAll('.prd-legend');
  assert.equal(legends.length, 1, 'one legend per project');
  assert.ok(view.value.rules.segments, 'the served view carries no segments rule');
  assert.equal(legends[0].title, view.value.rules.segments);
  assert.deepEqual([...legends[0].querySelectorAll('.prd-legend-item')].map(item => item.textContent), legend.map(item => item.label));

  const rows = [...root.querySelectorAll('.prd-row')];
  assert.equal(rows.length, view.value.prds.length);
  let empties = 0;
  for (const rowEl of rows) {
    const expected = barSegments(byKey.get(rowEl.dataset.key), legend);
    const segments = [...rowEl.querySelectorAll('.prd-bar .prd-bar-seg')];
    assert.equal(segments.length, 6, rowEl.dataset.key);
    assert.deepEqual(segments.map(seg => seg.dataset.segment), legend.map(item => item.segment));
    assert.deepEqual(segments.map(seg => seg.title), expected.map(seg => seg.title));
    assert.deepEqual(segments.map(seg => seg.classList.contains('prd-bar-seg-empty')), expected.map(seg => seg.count === 0));
    empties += expected.filter(seg => seg.count === 0).length;
  }
  assert.ok(empties > 0, 'the served rows hold no empty segment, so the empty class goes unchecked');
});

test('prd lanes #20: each row figure is titled with its served rule and row detail, and reads the helpers\' text', t => {
  const ctx = servedWindow(t, 'prd_lanes', 'prd_lanes_tasks');
  const { root, view } = mountLanes(ctx);
  const { readyCell } = ctx.helpers;
  const { rules } = view.value;
  const byKey = rowsByKey(view);
  expandTail(ctx.window, root);

  let overdue = 0;
  let actions = 0;
  for (const rowEl of root.querySelectorAll('.prd-row')) {
    const row = byKey.get(rowEl.dataset.key);
    const head = rowEl.querySelector('.prd-row-head');
    for (const [cell, title, text] of rowFigures(ctx.helpers, rules, row)) {
      assert.deepEqual(reading(head, cell), { title, text }, `${row.key} ${cell}`);
    }

    const ready = readyCell(row.ready);
    assert.equal(head.querySelector('.prd-ready').classList.contains('overdue'), ready.overdue, row.key);
    if (ready.overdue) assert.notEqual(reading(head, '.prd-ready').title, rules.ready, `${row.key}: the title leaves the red unexplained`);
    overdue += ready.overdue ? 1 : 0;

    const action = head.querySelector('.prd-action');
    if (row.next_action.kind === 'none') {
      assert.equal(action, null, `${row.key} has no next action to show`);
      continue;
    }
    actions += 1;
    assert.ok(action.classList.contains(`prd-action-${row.next_action.kind}`), action.className);
  }
  assert.ok(overdue > 0, 'no served row is overdue, so the overdue class goes unchecked');
  assert.ok(actions > 0, 'no served row has a next action');
});

test('prd lanes #20: a fresh view is headed by one pip counting its PRDs, titled with the lanes rule', t => {
  const ctx = servedWindow(t, 'prd_lanes', 'prd_lanes_tasks');
  const { root, view } = mountLanes(ctx);
  const { prds, rules } = view.value;
  const pip = headPip(root, '.prd-lanes-head');

  assert.equal(view.state, 'fresh');
  assert.equal(pip.text, `${prds.length} PRDs with open work`);
  assert.equal(pip.title, rules.lanes);
});

test('prd lanes #20: a lower-bound view marks only its head, and every figure keeps its rule and row data on hover', t => {
  const ctx = servedWindow(t, 'prd_lanes_queueless', 'prd_lanes_tasks');
  const { root, view } = mountLanes(ctx);
  const { LOWER_BOUND_PREFIX } = ctx.window.DF_DATUM;
  const { rules, lanes } = view.value;
  const byKey = rowsByKey(view);
  expandTail(ctx.window, root);

  assert.equal(view.state, 'lower_bound');
  const pip = headPip(root, '.prd-lanes-head');
  assert.ok(pip.text.startsWith(LOWER_BOUND_PREFIX), pip.text);
  assert.equal(pip.title, view.reason);

  assert.deepEqual(
    [...root.querySelectorAll('.prd-lane-head')].map(head => reading(head.parentElement, '.prd-lane-head')),
    lanes.map(entry => ({ title: entry.rule, text: `${entry.label} (${entry.prd_keys.length})` })),
  );
  const rows = [...root.querySelectorAll('.prd-row')];
  assert.equal(rows.length, view.value.prds.length);
  for (const rowEl of rows) {
    const row = byKey.get(rowEl.dataset.key);
    const head = rowEl.querySelector('.prd-row-head');
    for (const [cell, title, text] of rowFigures(ctx.helpers, rules, row)) {
      assert.deepEqual(reading(head, cell), { title, text }, `${row.key} ${cell}`);
    }
  }

  // The same served view restated fresh draws the same footer and strip text:
  // the lower bound qualifies the view once, at its head, never its figures.
  const fresh = mountLanes(ctx, { ...view, state: 'fresh', reason: null }).root;
  for (const cell of ['.dark-matter-head', '.prd-closed', '.prd-coverage']) {
    assert.deepEqual(reading(root, cell), reading(fresh, cell), cell);
  }

  const titledWithReason = [...root.querySelectorAll('[title]')].filter(el => el.title === view.reason);
  assert.deepEqual(titledWithReason, [pip.el], 'the reason must title the head pip alone');
});

test('prd lanes #20: the contention line renders every served top entry unsliced, and an unknown line says why', t => {
  const contended = servedWindow(t, 'prd_lanes_contended');
  const { contentionEntryText, heldShareText, neverStartedText, projectPrdContention } = contended.helpers;
  const { PrdContentionLine } = contended.window.DF_PRD_LANES;
  const contention = projectPrdContention(contended.D, PROJECT);
  const { value } = contention;
  const line = contended.harness.mount(PrdContentionLine, { contention }).container;

  assert.ok(value.top.length > 0);
  const pip = headPip(line, '.prd-contention-head');
  assert.deepEqual({ title: pip.title, text: pip.text }, { title: value.rules.top, text: `${value.top.length} most-wanted lock keys` });
  const entries = [...line.querySelectorAll('.prd-contention-entry')];
  assert.deepEqual(
    entries.map(entry => ({ title: entry.querySelector('[title]').title, text: entry.textContent })),
    value.top.map(entry => ({ title: value.rules.top, text: contentionEntryText(entry) })),
  );
  assert.deepEqual(reading(line, '.prd-held-share'), { title: value.rules.held_share, text: heldShareText(value.held_share) });
  assert.deepEqual(reading(line, '.prd-never-started'), { title: value.rules.never_started, text: neverStartedText(value.never_started) });

  const offline = servedWindow(t, 'prd_lanes_scheduler_offline');
  const unknown = offline.helpers.projectPrdContention(offline.D, PROJECT);
  assert.equal(unknown.state, 'unknown');
  const dark = offline.harness.mount(offline.window.DF_PRD_LANES.PrdContentionLine, { contention: unknown }).container;
  assert.equal(dark.querySelector('.prd-contention').textContent, unknown.reason);
  assert.equal(dark.querySelectorAll('.prd-contention-entry').length, 0);
});

test('prd lanes #20: the dark-matter strip shows the served total and expands to one group per served reason', t => {
  const ctx = servedWindow(t, 'prd_lanes', 'prd_lanes_tasks');
  const { root, view, pool, selected } = mountLanes(ctx);
  const { darkMatterSummary, darkMatterPage, membersDrawn, DARK_MATTER_PAGE_SIZE } = ctx.helpers;
  const { dark_matter: darkMatter, rules } = view.value;
  const strip = root.querySelector('.dark-matter');

  assert.deepEqual(reading(strip, '.dark-matter-head'), { title: rules.dark_matter, text: darkMatterSummary(darkMatter) });
  assert.ok(darkMatter.hidden_high.count > 0);
  const hidden = reading(strip, '.dark-matter-hidden-high');
  assert.equal(hidden.title, rules.hidden_high);
  for (const ref of darkMatter.hidden_high.refs) assert.ok(hidden.text.includes(`#${ref.task_id}`), hidden.text);

  assert.equal(strip.querySelectorAll('.dark-matter-group').length, 0, 'the members start collapsed');
  click(ctx.window, strip.querySelector('.dark-matter-toggle'));
  const groups = [...strip.querySelectorAll('.dark-matter-group')];
  assert.deepEqual(groups.map(group => group.dataset.reason), Object.keys(darkMatter.members));
  let drawn = 0;
  for (const group of groups) {
    const served = darkMatter.members[group.dataset.reason];
    const lines = [...group.querySelectorAll('.dark-matter-row')];
    const expected = membersDrawn(pool, PROJECT, darkMatterPage(served, 1).ids).rows;
    assert.ok(lines.length <= DARK_MATTER_PAGE_SIZE);
    assert.deepEqual(lines.map(line => line.dataset.rowId), expected.map(row => row.id));
    drawn += lines.length;
  }
  assert.ok(drawn > 0, 'no dark-matter member was drawn');

  const first = strip.querySelector('.dark-matter-row');
  click(ctx.window, first);
  assert.deepEqual(selected, [first.dataset.rowId]);
});

test('prd lanes #20: the footer renders the closed and coverage figures and one titled pip per served input', t => {
  const ctx = servedWindow(t, 'prd_lanes', 'prd_lanes_tasks');
  const { root, view } = mountLanes(ctx);
  const { inputPip } = ctx.helpers;
  const { closed, signal_coverage: coverage, multi_link: multiLink, inputs, rules } = view.value;
  const footer = root.querySelector('.prd-footer');

  const closedReading = reading(footer, '.prd-closed');
  assert.equal(closedReading.title, rules.closed);
  assert.ok(closedReading.text.startsWith(`${closed.prds} PRDs closed (${closed.tasks} tasks)`), closedReading.text);
  const coverageReading = reading(footer, '.prd-coverage');
  assert.equal(coverageReading.title, rules.signal_coverage);
  assert.ok(coverageReading.text.startsWith(`${coverage.named} of ${coverage.of} `), coverageReading.text);
  assert.equal(multiLink, 0);
  assert.equal(footer.querySelector('.prd-multi-link'), null, 'multi_link renders only when non-zero');

  const pips = [...footer.querySelectorAll('.pip')];
  assert.deepEqual(pips.map(pip => pip.dataset.input), inputs.map(input => input.name));
  assert.deepEqual(
    pips.map(pip => ({ title: pip.title, text: pip.textContent })),
    inputs.map(input => ({ title: inputPip(input).title, text: inputPip(input).text })),
  );
});

test('prd lanes #20: an expanded row draws its open members from the pool and lists the ids behind its figures', t => {
  const ctx = servedWindow(t, 'prd_lanes', 'prd_lanes_tasks');
  const { root, view, pool } = mountLanes(ctx);
  const { membersDrawn } = ctx.helpers;
  const { rules } = view.value;
  const byKey = rowsByKey(view);

  const subjects = view.value.prds.filter(row => row.lane !== 'tail' && (
    row.detail.blocker_refs.length || row.detail.oldest_ready_high_ids.length || row.detail.deferred_unrecorded_ids.length
  ));
  const checked = { blocker_refs: 0, oldest_ready_high_ids: 0, deferred_unrecorded_ids: 0, members: 0 };
  for (const subject of subjects) {
    const rowEl = root.querySelector(`.prd-row[data-key="${subject.key}"]`);
    assert.equal(rowEl.querySelector('.prd-row-body'), null, 'a row starts collapsed');
    click(ctx.window, rowEl.querySelector('.prd-row-head'));
    const body = rowEl.querySelector('.prd-row-body');
    const row = byKey.get(subject.key);

    assert.equal(reading(body, '.prd-activity').title, rules.activity);
    const ids = [...row.direct_open_ids, ...row.followup_open.map(followup => followup.task_id)];
    const { rows, notDrawn } = membersDrawn(pool, PROJECT, ids);
    assert.deepEqual([...body.querySelectorAll('.members-stub li')].map(li => li.textContent), rows.map(member => member.title));
    checked.members += rows.length;
    const notDrawnEl = body.querySelector('.prd-not-drawn');
    if (notDrawn.length === 0) assert.equal(notDrawnEl, null);
    else for (const id of notDrawn) assert.ok(notDrawnEl.textContent.includes(`#${id}`), notDrawnEl.textContent);

    const chips = selector => [...body.querySelectorAll(`${selector} .chip`)].map(chip => chip.textContent);
    const blockers = chips('.prd-blockers');
    assert.equal(blockers.length, row.detail.blocker_refs.length);
    row.detail.blocker_refs.forEach((ref, i) => {
      assert.ok(blockers[i].includes(`#${ref.task_id}`) && blockers[i].includes(ref.prd ?? 'no PRD'), blockers[i]);
    });
    assert.deepEqual(chips('.prd-ready-high'), row.detail.oldest_ready_high_ids.map(id => `#${id}`));
    assert.deepEqual(chips('.prd-deferred-unrecorded'), row.detail.deferred_unrecorded_ids.map(id => `#${id}`));
    for (const field of ['blocker_refs', 'oldest_ready_high_ids', 'deferred_unrecorded_ids']) {
      checked[field] += row.detail[field].length;
    }
  }
  for (const [what, n] of Object.entries(checked)) assert.ok(n > 0, `no expanded row exercised ${what}`);
});

test('prd lanes #20: an expanded row stays expanded when a poll moves it to another lane', t => {
  const ctx = servedWindow(t, 'prd_lanes', 'prd_lanes_tasks');
  const { root, view, handle, props } = mountLanes(ctx);
  const { lanes, prds } = view.value;
  const from = lanes.find(entry => entry.lane === 'stuck');
  const to = lanes.find(entry => entry.lane === 'moving');
  const [key] = from.prd_keys;
  const rowEl = () => root.querySelector(`.prd-row[data-key="${key}"]`);

  click(ctx.window, rowEl().querySelector('.prd-row-head'));
  assert.ok(rowEl().querySelector('.prd-row-body'), 'the click did not expand the row');

  const moved = {
    ...view,
    value: {
      ...view.value,
      lanes: lanes.map(entry => {
        if (entry === from) return { ...entry, prd_keys: entry.prd_keys.filter(each => each !== key) };
        if (entry === to) return { ...entry, prd_keys: [key, ...entry.prd_keys] };
        return entry;
      }),
      prds: prds.map(row => (row.key === key ? { ...row, lane: to.lane } : row)),
    },
  };
  handle.rerender({ ...props, view: moved });
  assert.equal(rowEl().closest('.prd-lane').dataset.lane, to.lane, 'the re-poll did not move the row');
  assert.ok(rowEl().querySelector('.prd-row-body'), 'the poll that moved the row closed it');
});

test('prd lanes #20: an unknown view renders its served reason in place of the lanes', t => {
  for (const name of ['prd_lanes_lineage_failed', 'prd_view_map_failed']) {
    const ctx = servedWindow(t, name);
    const { root, view } = mountLanes(ctx);
    assert.equal(view.state, 'unknown', name);
    assert.ok(view.reason, name);
    assert.equal(root.querySelector('.prd-unknown').textContent, view.reason, name);
    assert.equal(root.querySelectorAll('.prd-lane').length, 0, name);
    assert.equal(root.querySelector('.prd-legend'), null, name);
  }
});

// ── Row 21: the Tasks tab hosts the lanes page ──

const TAB_PROPS = { projectFilter: [], search: '' };
const GROUPING_KEY = 'df.tasksGroupByPrd';

// The real Tasks tab over both served projects, the /tasks body served beside
// prd_lanes and the /prds body `prdsName`, with `stored` written as the
// grouping preference before the mount, and every url the page fetches
// recorded.
function tasksTab(t, prdsName, { stored = null } = {}) {
  const ctx = servedWindow(t, prdsName, 'prd_lanes_tasks');
  const { window } = ctx;
  seedDashboardData(window, {
    PROJECTS: Object.keys(payload('prd_lanes_tasks').TASKS_SNAPSHOT).map(projectRow),
    __receipt: receiptsFor(window.DF_DATA_LOADER, ['PROJECTS']),
  });
  if (stored !== null) window.localStorage.setItem(GROUPING_KEY, JSON.stringify(stored));
  const requested = [];
  const unsettled = window.fetch;
  window.fetch = (url, init) => {
    requested.push(String(url));
    return unsettled(url, init);
  };
  const handle = ctx.harness.mount(window.DF_TASKS.TasksTab, TAB_PROPS);
  return { ...ctx, handle, container: handle.container, requested };
}

function projectPanel(container, project) {
  const head = [...container.querySelectorAll('.proj-head')].find(el => el.querySelector('.name')?.textContent === project);
  assert.ok(head, `no project group for ${project}`);
  return head.parentElement;
}

function groupingButton(panel, label) {
  const buttons = [...panel.querySelectorAll('.proj-head .seg button')].filter(button => button.textContent === label);
  assert.equal(buttons.length, 1, `expected one '${label}' grouping button`);
  return buttons[0];
}

function graphTitles(scope) {
  return [...scope.querySelectorAll('.node .ttl')].map(el => el.textContent);
}

function openIds(row) {
  return [...row.direct_open_ids, ...row.followup_open.map(followup => followup.task_id)];
}

test('prd lanes #21: a project with no stored grouping opens on the lanes, and nothing is stored', t => {
  const tab = tasksTab(t, 'prd_lanes');
  const panel = projectPanel(tab.container, PROJECT);

  assert.ok(groupingButton(panel, 'by PRD').classList.contains('on'), 'by PRD is not the default');
  assert.ok(panel.querySelectorAll('.prd-lane').length > 0, 'the default view draws no lanes');
  assert.equal(panel.querySelectorAll('.node').length, 0, 'a collapsed lanes page draws no graph');
  assert.equal(tab.window.localStorage.getItem(GROUPING_KEY), null, 'the default grouping was stored');

  // reify serves a lower-bound view with no rows: its head says so, and
  // nothing under it does.
  const reify = projectPanel(tab.container, 'reify');
  const view = tab.helpers.projectPrdView(tab.D, 'reify');
  const { lanes, closed } = view.value;
  assert.equal(view.state, 'lower_bound');
  assert.equal(view.value.prds.length, 0);
  assert.ok(headPip(reify, '.prd-lanes-head').text.startsWith(tab.window.DF_DATUM.LOWER_BOUND_PREFIX));
  assert.deepEqual(
    [...reify.querySelectorAll('.prd-lane-head')].map(head => head.textContent),
    lanes.map(entry => `${entry.label} (0)`),
  );
  assert.equal(reify.querySelector('.prd-closed').textContent, `${closed.prds} PRDs closed (${closed.tasks} tasks)`);
});

test('prd lanes #21: a project stored as listed flat opens on the graph, and by PRD removes the key', t => {
  const tab = tasksTab(t, 'prd_lanes', { stored: { [PROJECT]: 'flat' } });
  const panel = projectPanel(tab.container, PROJECT);

  assert.ok(groupingButton(panel, 'list').classList.contains('on'), 'the stored list view was not respected');
  assert.ok(panel.querySelectorAll('.node').length > 0, 'the list view drew no graph');
  assert.equal(panel.querySelectorAll('.prd-lane').length, 0);

  click(tab.window, groupingButton(panel, 'by PRD'));
  assert.ok(groupingButton(panel, 'by PRD').classList.contains('on'));
  assert.ok(panel.querySelectorAll('.prd-lane').length > 0);
  assert.equal(tab.window.localStorage.getItem(GROUPING_KEY), null, 'by PRD is the default, yet its key is still stored');
});

test('prd lanes #21: the lanes request no terminal window; only the Complete view does', async t => {
  const tab = tasksTab(t, 'prd_lanes');
  const flush = () => new Promise(resolve => setImmediate(resolve));
  const terminalRequests = () => tab.requested.filter(url => url.includes('terminal='));
  await flush();
  assert.deepEqual(terminalRequests(), [], 'the by-PRD view requested a terminal window');

  const { label } = tab.window.DF_TASK_SNAPSHOT.CENSUS_VIEWS.find(view => view.key === 'terminal');
  const [complete] = [...tab.container.querySelectorAll('.seg button')].filter(button => button.textContent === label);
  click(tab.window, complete);
  await flush();
  const forProject = terminalRequests().filter(url => url.endsWith(`?terminal=${PROJECT}`));
  assert.equal(forProject.length, 1, `the Complete view must request ${PROJECT}'s window once: ${terminalRequests().join(', ')}`);
});

test('prd lanes #21: the contention line heads the project in both modes', t => {
  const tab = tasksTab(t, 'prd_lanes');
  const panel = projectPanel(tab.container, PROJECT);
  const contention = tab.helpers.projectPrdContention(tab.D, PROJECT);
  const heldShare = () => {
    const lines = panel.querySelectorAll('.prd-contention');
    assert.equal(lines.length, 1, 'expected one contention line');
    return reading(lines[0], '.prd-held-share').text;
  };

  assert.equal(heldShare(), tab.helpers.heldShareText(contention.value.held_share));
  click(tab.window, groupingButton(panel, 'list'));
  assert.ok(panel.querySelectorAll('.node').length > 0);
  assert.equal(heldShare(), tab.helpers.heldShareText(contention.value.held_share));
});

test('prd lanes #21: an expanded row draws its open members as graph nodes from the narrowed rows', t => {
  const tab = tasksTab(t, 'prd_lanes');
  const { membersDrawn, projectPrdView } = tab.helpers;
  const pool = tab.window.DF_TASK_SNAPSHOT.projectRows(tab.D, PROJECT).value;
  const subject = projectPrdView(tab.D, PROJECT).value.prds.find(
    row => row.lane !== 'tail' && membersDrawn(pool, PROJECT, openIds(row)).rows.length >= 2,
  );
  assert.ok(subject, 'no served row has two drawn open members to narrow between');
  const members = membersDrawn(pool, PROJECT, openIds(subject)).rows;
  const panel = projectPanel(tab.container, PROJECT);
  const rowEl = () => panel.querySelector(`.prd-row[data-key="${subject.key}"]`);

  click(tab.window, rowEl().querySelector('.prd-row-head'));
  assert.deepEqual(graphTitles(rowEl()).sort(), members.map(row => row.title).sort());

  const [kept] = members;
  tab.handle.rerender({ ...TAB_PROPS, search: kept.title });
  assert.deepEqual(graphTitles(rowEl()), [kept.title], 'the search did not narrow the expanded graph');
  const { notDrawn } = membersDrawn(pool.filter(row => row.id === kept.id), PROJECT, openIds(subject));
  assert.ok(notDrawn.length > 0);
  const notDrawnText = rowEl().querySelector('.prd-not-drawn').textContent;
  for (const id of notDrawn) assert.ok(notDrawnText.includes(`#${id}`), notDrawnText);

  const renamed = `${kept.title} renamed`;
  const snapshot = structuredClone(payload('prd_lanes_tasks').TASKS_SNAPSHOT);
  snapshot[PROJECT].rows.value.find(row => row.id === kept.id).title = renamed;
  seedDashboardData(tab.window, { TASKS_SNAPSHOT: snapshot });
  tab.handle.rerender({ ...TAB_PROPS, search: kept.title });
  assert.deepEqual(graphTitles(rowEl()), [renamed], 'the expanded graph did not draw the latest poll\'s title');
});

test('prd lanes #21: focus on a member narrows the expanded row to its neighborhood, listing the rest as not drawn', t => {
  const tab = tasksTab(t, 'prd_lanes');
  const { membersDrawn, projectPrdView } = tab.helpers;
  const { focusGroupView } = tab.window.DF_GRAPH_LAYOUT;
  const pool = tab.window.DF_TASK_SNAPSHOT.projectRows(tab.D, PROJECT).value;
  const focusedOn = anchor => focusGroupView(pool, { focusMode: true, selectedId: anchor.id, focusAnchorId: anchor.id }).shown;
  // A row whose drawn members focus on one of them would narrow: some, not all.
  const subjects = projectPrdView(tab.D, PROJECT).value.prds.filter(row => row.lane !== 'tail').flatMap(row => {
    const members = membersDrawn(pool, PROJECT, openIds(row)).rows;
    return members.map(anchor => ({ row, members, anchor, focused: membersDrawn(focusedOn(anchor), PROJECT, openIds(row)) }));
  });
  const subject = subjects.find(({ members, focused }) => members.length >= 2 && focused.rows.length < members.length);
  assert.ok(subject, 'no served row has a member whose focus would leave another member out');
  const { row, anchor, focused } = subject;
  const panel = projectPanel(tab.container, PROJECT);
  const rowEl = () => panel.querySelector(`.prd-row[data-key="${row.key}"]`);

  click(tab.window, rowEl().querySelector('.prd-row-head'));
  const node = [...rowEl().querySelectorAll('.node')].find(el => el.querySelector('.ttl').textContent === anchor.title);
  assert.ok(node, `${anchor.id} is not drawn in the expanded row`);
  tab.window.React.act(() => node.dispatchEvent(new tab.window.MouseEvent('dblclick', { bubbles: true })));

  assert.ok(focused.rows.some(member => member.id === anchor.id));
  assert.deepEqual(graphTitles(rowEl()).sort(), focused.rows.map(member => member.title).sort(), 'focus did not narrow the expanded row');
  const notDrawnText = rowEl().querySelector('.prd-not-drawn').textContent;
  for (const id of focused.notDrawn) assert.ok(notDrawnText.includes(`#${id}`), notDrawnText);
});

test('prd lanes #21: an unknown view shows its served reason, and list still draws the graph', t => {
  const tab = tasksTab(t, 'prd_lanes_lineage_failed');
  const view = tab.helpers.projectPrdView(tab.D, PROJECT);
  const panel = projectPanel(tab.container, PROJECT);

  assert.equal(view.state, 'unknown');
  assert.equal(panel.querySelector('.prd-unknown').textContent, view.reason);
  click(tab.window, groupingButton(panel, 'list'));
  assert.ok(panel.querySelectorAll('.node').length > 0, 'the list view drew no graph');
});

