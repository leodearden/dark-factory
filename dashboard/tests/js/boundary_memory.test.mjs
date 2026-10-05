// Boundary family 4b, memory operations: sketch #11.
//
// The REAL /memory-graphs route served this body over a write journal holding
// reads, writes and one kind that is neither (test_boundary_js.py builds it).
// It lands through data.js's real refreshOne and is read the way the Overview
// caption and the Memory tab's caption and donut read it.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { applyServed, loadClient, payload } from './_boundary_client.mjs';

const MEMORY_GRAPHS = '/api/v2/dashboard/memory-graphs';
const RECEIVED_AT = Date.parse('2026-10-04T09:00:01Z');

test('sketch #11: reads, writes and other sum to the donut total, and every surface reads one opsTotals datum', async () => {
  const client = loadClient();
  assert.equal(await applyServed(client, MEMORY_GRAPHS, payload('memory_ops'), { receivedAt: RECEIVED_AT }), 'applied');
  const D = client.window.DF_DATA;
  const { opsTotals, opsCaption, opsTotalText } = client.window.DF_MEMORY_READINGS;

  const totals = opsTotals(D);
  assert.equal(totals.state, 'fresh');
  assert.equal(totals.value, opsTotals(D).value, 'the Overview and the Memory tab read the same served totals');
  const { reads, writes, other, total } = totals.value;
  assert.equal(reads + writes + other, total);
  assert.ok(other > 0, 'the kind outside read and write is counted');

  const caption = client.window.DF_DATUM.datumView(totals, { now: RECEIVED_AT, format: opsCaption }).text;
  assert.equal(caption, `${reads} reads · ${writes} writes · ${other} other`);
  assert.equal(opsTotalText(D), String(total), 'the donut centre');
  const slices = D.MEMORY_OPS.by_operation.reduce((sum, slice) => sum + slice.value, 0);
  assert.equal(slices, total, 'the donut slices are the same rows the centre counts');
});
