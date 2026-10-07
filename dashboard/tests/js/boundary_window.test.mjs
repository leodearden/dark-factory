// Boundary family 3, the window: sketch #8 and #9.
//
// Each test lands a body a REAL windowed route served (test_boundary_js.py
// builds them) through data.js's real refreshOne, so the receipt carries the
// served WINDOW echo exactly as a poll would, then reads it the way the
// window chip and the Merge tab do.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { applyServed, loadClient, payload } from './_boundary_client.mjs';

const RECEIVED_AT = Date.parse('2026-10-04T09:00:01Z');

async function landWindowed(endpoint, win, name) {
  const client = loadClient();
  const body = payload(name);
  const outcome = await applyServed(client, `${endpoint}?window=${win}`, body, { win, receivedAt: RECEIVED_AT });
  assert.equal(outcome, 'applied', name);
  return { client, body };
}

test('sketch #8: a declined 90d request labels the chip with the window served, lights no chip, and resets on a tab switch', async () => {
  const COSTS = '/api/v2/dashboard/costs';
  const { client, body } = await landWindowed(COSTS, '90d', 'costs_90d');
  const { windowEcho, windowLabel, highlightedWindow, windowForTab, TAB_WINDOWS, DEFAULT_WINDOW } = client.window.DF_WINDOW_CHIP;
  assert.equal(TAB_WINDOWS.cost.endpoint, COSTS);

  const echo = windowEcho(client.window.DF_DATA.__receipt, COSTS);
  assert.deepEqual(echo, body.WINDOW);
  assert.equal(windowLabel(echo), '30d (90d not available)');
  assert.equal(highlightedWindow(echo, TAB_WINDOWS.cost.windows), null);

  assert.equal(windowForTab('burn', '90d'), '90d', 'the burndown chip offers 90d');
  assert.equal(windowForTab('cost', '90d'), DEFAULT_WINDOW, 'switching to a tab without 90d resets the chip');
});

const MERGE_QUEUE = '/api/v2/dashboard/merge-queue';

async function mergeBlock(win) {
  const { client, body } = await landWindowed(MERGE_QUEUE, win, `merge_${win}`);
  const D = client.window.DF_DATA;
  const echo = client.window.DF_WINDOW_CHIP.windowEcho(D.__receipt, MERGE_QUEUE);
  return { client, body, block: D.MERGE_QUEUE['dark-factory'], echo };
}

const sum = values => values.reduce((total, value) => total + value, 0);
const DAY_MS = 24 * 60 * 60 * 1000;

test('sketch #9: the recent-merges caption counts only its own window\'s merges, and the latency caption splits every attempt', async () => {
  const totals = {};
  for (const win of ['7d', '24h']) {
    const { client, body, block, echo } = await mergeBlock(win);
    const { recentMergesCaption } = client.window.DF_WINDOW_CHIP;
    const { latencyCaption } = client.window.DF_MERGE_QUEUE;

    assert.equal(
      recentMergesCaption(block.recent.length, block.recent_total.value, echo),
      `showing ${block.recent.length} of ${block.recent_total.value} in ${win}`,
    );
    const opensAt = Date.parse(body.served_at) - echo.days * DAY_MS;
    for (const row of block.recent) {
      assert.ok(Date.parse(row.timestamp) >= opensAt, `${win} counts a merge from ${row.timestamp}, outside it`);
    }
    totals[win] = block.recent_total.value;

    const attempts = sum(block.outcomes.values);
    const timed = block.latency.with_duration;
    assert.equal(latencyCaption(block.latency), `of ${timed} with recorded duration · ${attempts - timed} without`);
  }
  assert.ok(totals['24h'] < totals['7d'], `24h holds ${totals['24h']} merges and 7d ${totals['7d']}`);
});

test('sketch #9: the donut centre, summed inline in tabs.jsx and so asserted here as arithmetic over the served block, counts each attempt once', async () => {
  for (const win of ['7d', '24h']) {
    const { block } = await mergeBlock(win);
    assert.equal(sum(block.outcomes.values), block.latency.with_duration + block.latency.without_duration, win);
    assert.equal(block.outcomes.values.length, block.outcomes.labels.length, win);
  }
});
