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
import { servingEndpoint } from './wire_fixtures.mjs';

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

// ProjectPrdLanes over the window's served view and rows, with a member
// renderer that draws one li per row title, and every onSelect recorded.
function mountLanes(ctx) {
  const { window, D, helpers } = ctx;
  const { React } = window;
  const view = helpers.projectPrdView(D, PROJECT);
  const pool = window.DF_TASK_SNAPSHOT.projectRows(D, PROJECT).value || [];
  const selected = [];
  const renderMembers = rows => React.createElement(
    'ul', { className: 'members-stub' }, rows.map(row => React.createElement('li', { key: row.id }, row.title)),
  );
  const handle = ctx.harness.mount(window.DF_PRD_LANES.ProjectPrdLanes, {
    project: PROJECT, view, pool, renderMembers, onSelect: id => selected.push(id),
  });
  return { root: handle.container, view, pool, selected };
}

function rowsByKey(view) {
  return new Map(view.value.prds.map(row => [row.key, row]));
}

function expandTail(window, root) {
  click(window, root.querySelector('.prd-lane[data-lane="tail"] .prd-tail-line'));
}

// The title and the text of the one Datum reading inside `scope`'s `selector`.
function reading(scope, selector) {
  const el = scope.querySelector(`${selector} [title]`);
  assert.ok(el, `no titled reading under ${selector}`);
  return { title: el.title, text: el.textContent };
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
  assert.ok(line[0].textContent.startsWith(`${entry.prd_keys.length} `), line[0].textContent);
  for (const key of entry.prd_keys) {
    assert.ok(line[0].textContent.includes(byKey.get(key).title), `${key}'s title is missing from the tail line`);
  }

  click(ctx.window, line[0]);
  assert.deepEqual([...tail.querySelectorAll('.prd-row')].map(row => row.dataset.key), entry.prd_keys);
});

test('prd lanes #20: every row bar draws the six served segments in legend order, under one legend', t => {
  const ctx = servedWindow(t, 'prd_lanes', 'prd_lanes_tasks');
  const { root, view } = mountLanes(ctx);
  const { barSegments } = ctx.helpers;
  const legend = view.value.segment_legend;
  const byKey = rowsByKey(view);
  expandTail(ctx.window, root);

  const legends = root.querySelectorAll('.prd-legend');
  assert.equal(legends.length, 1, 'one legend per project');
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

test('prd lanes #20: each row figure is titled with its served rule and reads the helpers\' text', t => {
  const ctx = servedWindow(t, 'prd_lanes', 'prd_lanes_tasks');
  const { root, view } = mountLanes(ctx);
  const { readyCell, doneOfTotal, signalCell } = ctx.helpers;
  const { rules } = view.value;
  const byKey = rowsByKey(view);
  expandTail(ctx.window, root);

  let overdue = 0;
  let actions = 0;
  for (const rowEl of root.querySelectorAll('.prd-row')) {
    const row = byKey.get(rowEl.dataset.key);
    const head = rowEl.querySelector('.prd-row-head');

    const title = reading(head, '.prd-title');
    assert.ok(title.title.startsWith(row.key) && title.title.endsWith(rules.title), title.title);
    assert.equal(title.text, row.title);

    assert.deepEqual(reading(head, '.prd-state'), { title: rules.state_line, text: row.state_line });

    const total = doneOfTotal(row);
    const totalReading = reading(head, '.prd-total');
    assert.equal(totalReading.text, total.text);
    assert.ok(totalReading.title.includes(total.detail) && totalReading.title.endsWith(rules.total), totalReading.title);

    const signal = signalCell(row.signal);
    const signalReading = reading(head, '.prd-signal');
    assert.equal(signalReading.text, signal.text);
    assert.ok(signalReading.title.endsWith(rules.signal), signalReading.title);
    if (signal.detail) assert.ok(signalReading.title.startsWith(signal.detail), signalReading.title);

    const ready = readyCell(row.ready);
    const readyEl = head.querySelector('.prd-ready');
    assert.deepEqual(reading(head, '.prd-ready'), { title: rules.ready, text: ready.text });
    assert.equal(readyEl.classList.contains('overdue'), ready.overdue, row.key);
    overdue += ready.overdue ? 1 : 0;

    const action = head.querySelector('.prd-action');
    if (row.next_action.kind === 'none') {
      assert.equal(action, null, `${row.key} has no next action to show`);
      continue;
    }
    actions += 1;
    assert.ok(action.classList.contains(`prd-action-${row.next_action.kind}`), action.className);
    assert.deepEqual(reading(head, '.prd-action'), { title: rules.next_action, text: row.next_action.text });
  }
  assert.ok(overdue > 0, 'no served row is overdue, so the overdue class goes unchecked');
  assert.ok(actions > 0, 'no served row has a next action');
});

test('prd lanes #20: the contention line renders every served top entry unsliced, and an unknown line says why', t => {
  const contended = servedWindow(t, 'prd_lanes_contended');
  const { contentionEntryText, heldShareText, neverStartedText, projectPrdContention } = contended.helpers;
  const { PrdContentionLine } = contended.window.DF_PRD_LANES;
  const contention = projectPrdContention(contended.D, PROJECT);
  const { value } = contention;
  const line = contended.harness.mount(PrdContentionLine, { contention }).container;

  assert.ok(value.top.length > 0);
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
  const { countsText, darkMatterPage, membersDrawn, DARK_MATTER_PAGE_SIZE } = ctx.helpers;
  const { dark_matter: darkMatter, rules } = view.value;
  const strip = root.querySelector('.dark-matter');

  const counts = reading(strip, '.dark-matter-head');
  assert.equal(counts.title, rules.dark_matter);
  assert.ok(counts.text.startsWith(`${darkMatter.total} `), counts.text);
  for (const by of ['by_status', 'by_priority', 'by_source', 'by_reason']) {
    assert.ok(counts.text.includes(countsText(darkMatter[by])), `${by}: ${counts.text}`);
  }
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
    for (const line of lines) assert.ok(served.includes(Number(line.dataset.taskId)), line.dataset.taskId);
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

