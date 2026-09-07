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

import contextlib
import errno
import fcntl
import json
import os
import sys
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
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


def _lease_filename(owner: str) -> str:
    """A filename no other holder can collide with, readable in a listing.

    Uniqueness is what lets the acquisition use ``O_CREAT | O_EXCL`` with no
    retry loop: holders never contend for a path, so an ``EEXIST`` would be a
    real defect rather than ordinary contention.  It also means no new holder
    can ever reuse a DEAD holder's path, which is what makes reaping dead
    lease files safe (see :func:`reap_dead_leases`).

    The owner is slugified rather than dropped because these names are read
    by a human in cron mail; the pid and a uuid4 carry the uniqueness.
    """
    slug = ''.join(c if (c.isalnum() or c in '._-') else '-' for c in owner)
    slug = slug.strip('-')[:60] or 'holder'
    return f'{slug}.{os.getpid()}.{uuid.uuid4().hex}.lease'


def _lease_body(owner: str) -> bytes:
    """The diagnostics a prober reads out of a held lease file.

    Nothing branches on this — liveness is the flock, never the body — so a
    corrupt or empty body may degrade the diagnostic and must never degrade
    the guard.  Deliberately no expiry or TTL field: the kernel releases the
    flock when the holder dies, SIGKILL included, so there is no stale-lease
    case for a timeout to bound.
    """
    record = {
        'owner': owner,
        'pid': os.getpid(),
        'started_at': datetime.now(UTC).isoformat(),
    }
    return json.dumps(record).encode()


@contextlib.contextmanager
def hold_lease(owner: str, *, directory: Path | None = None) -> Iterator[bool]:
    """Publish a lease that holds :func:`main`'s sweep off while it is open.

    Yields whether the lease is HELD.  Take one around anything that seeds
    collections under :data:`PREFIXES` — the 6-hourly cron is otherwise free
    to delete a live run's corpus between its seed and measure phases.

    The exclusion is an ``fcntl.flock``, not the file's existence.  That is
    the whole reason there is no TTL, no expiry field and no pid-liveness
    probe: the kernel frees a flock when the holder dies, SIGKILL included,
    with no daemon, canary or ``atexit`` involved.  A TTL would force a trade
    with no good value — long enough for the longest live run, short enough
    that a crashed run does not wedge the cron — and a pid probe re-opens the
    PID-reuse hazard that makes a dead owner look alive.

    The flock is taken BEFORE the body is written, so a prober can never read
    a half-written record out of an unlocked file.  Each holder gets its own
    uniquely-named file, so concurrent holders (xdist workers, a bake-off
    beside an integration test) never contend and one release never
    un-guards another.

    FAILS OPEN — do not "tighten" this into a raise.  An unusable lease
    directory (unwritable, full, a regular file in the way) yields ``False``
    and reports one line on stderr instead of raising.  The callers are
    integration tests and two seeding scripts: raising would abort a live
    bake-off for a reason with nothing to do with what it measures, and fail
    an integration test on infrastructure noise.  What this guard improves
    on is "no guard at all", so degrading back to it is not a regression —
    degrading back to it SILENTLY would be, which is what the stderr line is
    for.  Same yielded-``held`` contract as
    ``shared/verify_admission.py::acquire_task_slot`` (clause C-fail-open).
    """
    target = lease_dir() if directory is None else Path(directory)
    path = target / _lease_filename(owner)
    fd = None
    # `held` is set only after the WHOLE acquisition succeeds, and is never
    # inferred from `fd`: a file opened but not flocked is not a lease, and
    # reporting one as held would be the silent no-guard this fails open to
    # avoid — the caller would believe it was covered while `live_leases`
    # correctly saw nothing.
    held = False
    try:
        target.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.set_inheritable(fd, False)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(fd, _lease_body(owner))
        held = True
    except OSError as exc:
        # Both facts on one line: the directory says WHERE to look, the
        # error says what to fix.
        print(
            f'Could not take an ephemeral-collection lease in {target} '
            f'({exc}); the cleanup sweep is NOT held off for {owner}',
            file=sys.stderr,
        )

    try:
        yield held
    finally:
        # Closing the fd is what releases the flock; the unlink is only
        # tidiness, and a holder that dies before reaching it leaves a file
        # that `reap_dead_leases` collects rather than a lease that holds
        # anything off.  Both run even for a PARTIAL acquisition — a file
        # created but not flocked is still this holder's litter.
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)


def _is_held(path: Path) -> bool | None:
    """Is some living process holding *path*'s flock?

    ``True`` a holder is live, ``False`` the holder is gone (or there never
    was one), ``None`` the question could not be asked at all — the file
    vanished under us, or is unreadable.

    The probe is ``LOCK_SH``, not ``LOCK_EX``: two sweepers running at once
    must not exclude each other and mistake a peer's probe for a live run.
    Shape copied from
    ``fused_memory/middleware/ticket_janitor.py::_orchestrator_running``;
    copied rather than imported because cron runs this file under the system
    ``python3``, where ``fused_memory`` is not importable.
    """
    try:
        handle = path.open('rb')
    except OSError:
        return None
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return True
            return None
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        handle.close()


def _describe(path: Path) -> dict:
    """Read a held lease's diagnostics, falling back to its filename.

    Nothing branches on the result: the guard has already decided the lease
    is live before this is called.  A holder killed between creating its
    file and writing its record therefore degrades what an operator reads in
    cron mail, and never whether the sweep holds off.
    """
    record = {'owner': path.name, 'pid': None, 'path': str(path)}
    try:
        body = json.loads(path.read_text())
    except (OSError, ValueError):
        return record
    if isinstance(body, dict):
        record['owner'] = body.get('owner') or path.name
        record['pid'] = body.get('pid')
    return record


def live_leases(*, directory: Path | None = None) -> list[dict]:
    """Return one record per lease a LIVING process is holding.

    Liveness is the flock and nothing else — not the file's existence, not
    an age, not a pid.  A lease file left behind by a SIGKILLed holder is
    unlocked the moment that process dies, so it is reported here as absent
    and collected later by :func:`reap_dead_leases`.

    Never raises.  A missing directory, a file that vanishes mid-scan, an
    unreadable file and an unparseable body are each ordinary rather than
    exceptional: this is called from an unattended cron job whose contract
    is "Always exits 0 (idempotent)".
    """
    target = lease_dir() if directory is None else Path(directory)
    try:
        entries = sorted(target.iterdir())
    except OSError:
        return []

    held = []
    for path in entries:
        if _is_held(path) is True:
            held.append(_describe(path))
    return held


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
