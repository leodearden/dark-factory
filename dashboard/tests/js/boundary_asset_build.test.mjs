// Boundary family: the serving asset build.
//
// The REAL /memory-graphs route served this body and its headers
// (test_boundary_js.py builds them, with the identities the app holds). It
// lands through data.js's real refreshOne, so the receipt carries the build
// identities exactly as a poll would, and the header names are pinned across
// Python and JS on the real wire.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { applyServed, loadClient, payload } from './_boundary_client.mjs';

const RECEIVED_AT = Date.parse('2026-11-18T09:00:01Z');

test('asset build: the receipt carries the build identities the server stamped on its response', async () => {
  const client = loadClient();
  const served = payload('asset_build_served');

  const outcome = await applyServed(client, served.path, served.body, { headers: served.headers, receivedAt: RECEIVED_AT });

  assert.equal(outcome, 'applied');
  assert.deepEqual(client.window.DF_DATA.__receipt[served.path].assetBuild, served.expected);
});
