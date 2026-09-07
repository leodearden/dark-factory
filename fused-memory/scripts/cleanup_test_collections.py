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

WHY THERE IS NO STORE-MUTATION PREFLIGHT HERE (an observation, task 4293)
------------------------------------------------------------------------
``fused_memory.utils.store_mutation_preflight.assert_store_mutation_allowed``
is the fail-closed capability probe the shared-store mutators in
``fused-memory/scripts/`` call before their first write.  The
``client.delete_collection`` below is unconditional and unprobed, and that is
a decision rather than an omission.  Three reasons, each measured against
this file AS IT STANDS at task 4293 — not a standing exemption for whatever
it becomes:

  * its blast radius is already bounded, statically.  Only names starting
    with a member of :data:`PREFIXES` are deleted, and the production
    ``collection_prefix`` default (``fused``) cannot match one — see the
    section above.  Bounding an otherwise unbounded mutation is exactly what
    the preflight is for; here an equality-pinned allowlist does it;
  * it never constructs a ``MemoryService``.  It talks to a raw
    ``QdrantClient`` and writes no mem0 SQLite history at all, so the
    capability the preflight probes for — writing mem0's history directory —
    is not one this script needs, and a probe would refuse runs that would
    have worked;
  * it is an unattended cron job whose contract at the top of this docstring
    is "Always exits 0 (idempotent)".  Every failure path here is a bare
    ``return``; there is no ``sys.exit`` in the file.  The preflight refuses
    by RAISING, deliberately (a refusal must not be mistakable for a handled
    outcome), which would break that contract.

If a later edit gives this script a ``MemoryService``, or lets any input
widen :data:`PREFIXES` past the two test-only names, the first two premises
die and the guard becomes required.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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


#: Environment override for :func:`lease_dir`.  It exists so tests never
#: write into the real directory the live cron reads — a test that left a
#: lease there would hold a real sweep off this host, and one that reaped
#: there would unlink a live run's lease.  An OPERATOR who sets it has to
#: set it for BOTH sides: a holder and a reaper that disagree about the
#: directory are exactly the silent-no-guard failure described below.
LEASE_DIR_ENV = 'DF_EPHEMERAL_COLLECTION_LEASE_DIR'

#: Where in-use leases live.  Hardcoded and absolute on purpose; see
#: :func:`lease_dir` for why every "portable" alternative is wrong here.
DEFAULT_LEASE_DIR = Path('/tmp/dark-factory-ephemeral-collection-leases')


def lease_dir() -> Path:
    """Return the directory in-use leases live in, resolved at CALL time.

    A hardcoded absolute ``/tmp`` path looks like a smell and is the one
    property that makes the guard correct.  The guard fires only if the
    HOLDER and the REAPER resolve the SAME directory, and they never share
    an environment: the reaper is this file, run by cron under a bare
    ``python3`` with a near-empty env; a holder is a pytest process (usually
    inside a ``.worktrees/<id>`` checkout, under an xdist worker) or a
    hand-run script in a login shell.

      * ``tempfile.gettempdir()`` honours ``TMPDIR``/``TEMP``/``TMP``, and
        pytest and cron set those differently;
      * ``XDG_RUNTIME_DIR`` is ``/run/user/<uid>`` in a login session and
        unset under cron;
      * a repo-relative path gives every worktree its own private lease
        directory even though ONE Qdrant at ``localhost:6333`` is shared by
        all of them — and it would leave the machine-operated
        ``project_root`` checkout dirty besides.

    Under any of those the reaper would read an empty directory, find no
    lease and delete: a guard that silently never fires, which is strictly
    worse than no guard because it also stops the next person looking.

    Resolved on every call, never memoised, so :data:`LEASE_DIR_ENV` can be
    redirected after this module is imported — which is what lets the test
    suite keep its hands off the real directory.  An empty value counts as
    unset: ``Path('')`` is the current working directory, i.e. precisely the
    repo-relative failure above.
    """
    override = os.environ.get(LEASE_DIR_ENV)
    return Path(override) if override else DEFAULT_LEASE_DIR


def main() -> None:
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        print('qdrant-client not installed, skipping', file=sys.stderr)
        return

    try:
        client = QdrantClient(url=QDRANT_URL, timeout=5)
    except Exception as exc:
        print(f'Qdrant unreachable ({exc}), skipping', file=sys.stderr)
        return

    # From here a client object EXISTS, so every exit path has to run through
    # the `finally` below.  The listing therefore lives inside it rather than
    # beside the construct: a Qdrant that accepts the TCP connection and then
    # times out (or 500s) on `get_collections` is the realistic unattended-cron
    # failure, and pairing the two in one earlier `try` returned from that
    # branch with the connection still open — bypassing the very `finally`
    # that exists to close it.
    deleted = 0
    try:
        try:
            collections = client.get_collections().collections
        except Exception as exc:
            print(f'Qdrant unreachable ({exc}), skipping', file=sys.stderr)
            return

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
