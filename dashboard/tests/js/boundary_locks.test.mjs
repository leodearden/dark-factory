// Boundary family 4c, locks: sketch #12.
//
// The REAL /scheduler route served a snapshot whose fan-out failed for P, and
// the REAL /tasks route served P's in-progress rows anyway (test_boundary_js.py
// builds both). They land through data.js's real refreshOne and each row's
// Locks cell is decided the way tabs.jsx::LocksCell decides it.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { applyServed, loadClient, payload } from './_boundary_client.mjs';

const SCHEDULER = '/api/v2/dashboard/scheduler';
const TASKS = '/api/v2/dashboard/tasks';
const OFFLINE = 'P';
const HEALTHY = 'H';

// LocksCell's lockInfo when no scheduler row joins: locksCellState decides on
// the Datum alone, so the cell under test is the one every row would draw.
const lockInfoFor = row => ({ rawTaskId: String(row.id).split('/T-').pop(), lockSet: [], moduleByPath: new Map() });

test('sketch #12: every in-progress row of a project whose scheduler is offline shows the placeholder and why, and no other project is blanked', async () => {
  const client = loadClient();
  const tasks = payload('tasks_with_p');
  const receivedAt = Date.parse(tasks.served_at);
  assert.equal(await applyServed(client, SCHEDULER, payload('scheduler_offline'), { receivedAt }), 'applied');
  assert.equal(await applyServed(client, TASKS, tasks, { receivedAt }), 'applied');
  const D = client.window.DF_DATA;
  const { locksCellState, schedulerLocksDatum } = client.window.DF_TASK_ROW_CELLS;
  const { projectRows } = client.window.DF_TASK_SNAPSHOT;
  const cellsOf = project => projectRows(D, project).value
    .filter(row => row.status === 'in-progress')
    .map(row => locksCellState(schedulerLocksDatum(D.SCHEDULER, project, D.__receipt), lockInfoFor(row)));

  const offline = cellsOf(OFFLINE);
  assert.ok(offline.length >= 2, `${OFFLINE} serves ${offline.length} in-progress rows`);
  for (const cell of offline) {
    assert.equal(cell.placeholder, '—');
    assert.match(cell.title, new RegExp(`${OFFLINE} offline`));
  }

  const healthy = cellsOf(HEALTHY);
  assert.ok(healthy.length >= 1);
  for (const cell of healthy) assert.deepEqual(cell, { placeholder: null, title: null });
});
