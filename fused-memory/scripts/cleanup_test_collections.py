#!/usr/bin/env python3
"""Delete stale ephemeral test collections from Qdrant.

Intended to run as a cron job.  Always exits 0 (idempotent).

WHAT IS AND IS NOT REAPABLE
---------------------------
A mem0 collection is named ``f'{collection_prefix}_{project_id}'``
(``Scope.mem0_collection_name``).  The DEFAULT ``collection_prefix`` is
``'fused'``, so nothing written by the running system matches any prefix
below — that is the design, not an oversight: this sweep must never be able
to delete live project memory.

A collection only becomes reapable because an integration test (or the E2
bake-off) OVERRIDES ``config.mem0.collection_prefix``.  Overriding only the
``project_id`` is the mistake to watch for: the collection then lands under
``fused`` and leaks permanently, because nothing reaps it and nobody looks.

Every prefix this script deletes is listed in :data:`PREFIXES`, and any
script that seeds under one imports its constant from HERE.  A name coined
in one file and reaped by a constant in another is one rename away from
orphaning collections forever.
"""

from __future__ import annotations

import sys

PREFIX = '_test_mem0_qdrant_integration_'

#: Prefix for the ephemeral collections the E2 storage-shape bake-off seeds
#: (``scripts/bake_off_storage_shape.py``, task 3199).  It lives HERE, in the
#: reaper, and is imported from here by the bake-off — a collection whose name
#: is coined in one file and reaped by another is one rename away from leaking
#: forever, and nothing under the default ``fused`` prefix is reapable by
#: design.
E2_BAKEOFF_PREFIX = '_test_e2_bakeoff'

#: Every prefix this sweep is allowed to delete.  Pinned by equality in the
#: tests: this tuple is the complete blast radius of a cron job that runs
#: unattended against the live Qdrant.
PREFIXES: tuple[str, ...] = (PREFIX, E2_BAKEOFF_PREFIX)

QDRANT_URL = 'http://localhost:6333'


def main() -> None:
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        print('qdrant-client not installed, skipping', file=sys.stderr)
        return

    try:
        client = QdrantClient(url=QDRANT_URL, timeout=5)
        collections = client.get_collections().collections
    except Exception as exc:
        print(f'Qdrant unreachable ({exc}), skipping', file=sys.stderr)
        return

    deleted = 0
    try:
        for col in collections:
            if not col.name.startswith(PREFIXES):
                continue
            try:
                client.delete_collection(col.name)
                deleted += 1
            except Exception as exc:
                # Reported, never fatal: aborting on the first stuck
                # collection would leave the rest of the leak in place, and
                # the whole point of the sweep is that nothing survives it.
                print(f'Failed to delete {col.name}: {exc}', file=sys.stderr)

        if deleted:
            print(f'Deleted {deleted} stale test collection(s)')
    finally:
        client.close()


if __name__ == '__main__':
    main()
