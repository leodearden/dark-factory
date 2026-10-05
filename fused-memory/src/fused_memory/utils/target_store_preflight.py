"""Fail-closed preflight against MIS-TARGETING a task store or escalation queue (tasks 4319, 5468).

THE POLICY THIS ENFORCES
------------------------
**An in-process READER OR MUTATOR of ``tasks.db`` or of the durable escalation
queue must refuse unless its target store already HOLDS A RECORD — merely
existing is not enough.** Neither substrate will tell you it was not there:
both silently create themselves and then report empty, so "nothing to do" and
"you are pointed at the wrong filesystem location" are the same output — and
the empty store that creation leaves behind EXISTS for every later run.

READERS are in scope, not merely mutators, and that is not a widening of
convenience: a read-only audit reporting "0 offenders" against a store it just
conjured is exactly as false as a no-op apply, and it is the read paths — dry
runs, CI predicates — that actually get pointed at the wrong checkout. Every
call site this guard has is reached on a READ path — and, where the script has
an apply mode at all, on that path too. One is a read-only CI/runbook
predicate with no ``--apply`` to gate on.

THE TWO MEASURED AUTO-CREATES (task 4319, re-measured at base 56fb6fec97)
-------------------------------------------------------------------------
* ``escalation/src/escalation/queue.py::EscalationQueue.__init__`` calls
  ``mkdir(parents=True, exist_ok=True)`` on its ``queue_dir``. Construct one on
  a missing path and the whole tree is created; ``get_pending()`` then returns
  ``[]``. Both escalation scripts here default ``--queue-dir`` to the RELATIVE
  ``./data/reconciliation/escalations`` and pass it through unresolved, so a run
  from anywhere but the project root — a task worktree in particular, which has
  no ``data/reconciliation/`` — manufactures an empty queue, reports
  ``"pending_before": 0``, and exits 0.
* ``fused_memory/backends/sqlite_task_backend.py::SqliteTaskBackend.get_tasks``
  auto-creates ``.taskmaster/tasks/tasks.db`` (see
  ``sqlite_task_backend.py::SqliteTaskBackend._get_connection``, which opens the
  file "creating parent directories") and returns ``{"tasks": []}`` for ANY
  ``project_root``, never raising. ``.taskmaster/`` is neither present in nor
  tracked by a worktree, so ``--project-root <worktree>`` yields an empty task
  tree, an empty plan, zero mutations, exit 0.

Both are a false all-clear indistinguishable from a clean one — the INV-11
``no-silent-fail-soft`` shape.

WHY THIS IS NOT ``assert_store_mutation_allowed``
--------------------------------------------------
``fused_memory/utils/store_mutation_preflight.py::assert_store_mutation_allowed``
exists for ONE failure mode: a mutation torn across TWO substrates under
OPPOSITE confinement (a Qdrant delete over the network, which landlock cannot
block, paired with a mem0 SQLite history write, which it can). Neither class
here has a second substrate, so neither can tear:

* ``sqlite_task_backend.py::SqliteTaskBackend``'s five mutating methods
  (``set_task_status``, ``set_status_and_stamp_audit``, ``update_task``,
  ``add_dependency``, ``remove_dependency``) all enter via
  ``async with self._write_lock(...), self._txn(...)``, and
  ``sqlite_task_backend.py::SqliteTaskBackend._txn`` is ONE explicit
  BEGIN/COMMIT with rollback-and-re-raise. ``update_task``'s three statements
  live inside that single transaction;
  ``sqlite_task_backend.py::SqliteTaskBackend.set_status_and_stamp_audit``
  states the property in its own docstring — status and both audit trails
  "both commit or both roll back". The module imports no HTTP, socket or
  subprocess client at all.
* In the escalation queue the FIRST write-requiring syscall of every mutating
  path is the lockfile ``os.open(..., O_CREAT | O_RDWR)`` in
  ``escalation/src/escalation/queue.py::escalation_id_lock`` — taken outside
  any handler, before the record is even read. A denial therefore aborts on
  record #1 with nothing mutated. The queue module imports no network client,
  and a script-constructed ``EscalationQueue`` has neither notify nor resolve
  callback wired (the sole wiring site is the orchestrator harness).

So the mem0-shaped guard is correctly ABSENT from both classes. This module is
its sibling, not its extension: different failure mode, different guard,
different population. The scope boundary between the two is recorded in
``store_mutation_preflight.py``'s own SCOPE section.

WHY NOT A CAPABILITY PROBE EITHER
----------------------------------
A write probe would be not merely low-value here but ACTIVELY MISLEADING: in
the scenario that actually happens the target directory is WRITABLE, so the
probe PASSES exactly when the danger is present. A guard that passes precisely
when the danger is present is worse than no guard. And where the target IS the
live store and writes ARE denied, the natural failure is already loud and
non-destructive — a ``PermissionError`` out of the queue lockfile's
``os.open``, or an ``OperationalError`` at connection-open, since WAL needs a
writable ``-shm``/``-wal`` beside ``tasks.db``. A probe would buy only a nicer
message, at the cost of creating a scratch file inside live production state.
Hence: a READ-ONLY check that the target already holds a record, which never
creates the store.

WHAT COUNTS AS POPULATED (task 5468)
------------------------------------
* Queue: the directory holds at least one entry. The residue is the empty
  directory ``EscalationQueue.__init__`` mkdirs; a live queue never returns to
  empty, because a resolve leaves ``archive/`` and lock sidecars behind.
* Task store: a ``tasks`` table with at least one row, read through a
  ``mode=ro`` URI so the probe cannot create the file. The residues are a
  zero-byte file and the row-less db ``get_tasks`` seeds.

MISSING and EMPTY are refused. An UNREADABLE target (a file at the queue path,
non-SQLite bytes or a directory at tasks.db) fails OPEN, because the backend's
own open already fails loudly.

RESIDUAL HOLES, left open deliberately:

* a POPULATED wrong location (another deployment's queue, another project's
  tasks.db) passes, since nothing pins the live store (see below);
* a stray file makes a residue queue directory pass;
* a read-only probe of a closed WAL-mode tasks.db may leave ``-shm``/``-wal``
  sidecars beside it.

PRIOR ART
---------
Task 2738 found the same ``get_tasks`` auto-create from a different consumer
and reached the same reading — see
``fused_memory/reconciliation/stages/task_knowledge_sync.py``, which records
that an auto-created empty store is "a false census, not a genuinely empty
project -- indistinguishable from one at the data layer", and gates on
``fused_memory/models/scope.py::resolve_main_checkout`` (an existence-style
check driving an ``unavailable`` sentinel) rather than on a capability probe.
That this task's answer matches a finding reached independently is why it is a
corroborated result rather than a fresh assertion.

TWO PLACEMENT RULES, which are the non-obvious part
----------------------------------------------------
* BEFORE THE SCAN, not at the ``--apply`` gate. The dry-run report is exactly
  as false as a no-op apply — ``"pending_before": 0`` from a manufactured queue
  IS the lie — and ``EscalationQueue.__init__`` performs its ``mkdir`` before
  ``apply`` is ever consulted, so an apply-gated check would fire only after
  the litter already exists. Putting it in ``run()``/``_run()`` rather than
  ``main()`` also means programmatic callers inherit it.
* A refusal RAISES; it never returns a report-shaped refusal. Here that is
  load-bearing rather than stylistic:
  ``scripts/dismiss_recon_integrity_noise.py::main`` and
  ``scripts/backfill_recon_escalations.py::main`` both ``return 0``
  UNCONDITIONALLY, with no error accounting — so a refusal routed through the
  normal report path would exit 0, reproducing the very defect this guard
  exists to fix.

  WHAT THAT BUYS, EXACTLY — and what it does not. The property secured is "a
  refusal is never 0". It is NOT "a refusal is distinguishable from every
  other exit code": each script ends in ``sys.exit(main())``, and wherever
  ``main()`` merely returns ``asyncio.run(_run(args))`` an uncaught
  ``TargetStoreMissing`` propagates and CPython exits **1** — which collides
  with the exit-1 rung the three task-store scripts already use
  (``audit_duplicate_tasks``'s "apply errors",
  ``correct_found_on_main_backlog``'s task-1175 "a write did not persist",
  ``audit_found_on_main_provenance``'s ``--fail-on-findings`` middle rung).
  A caller that needs the two told apart must catch
  ``TargetStoreMissing`` and map it onto a reserved code of its own; do not
  assume the traceback is self-identifying to an exit-code-only consumer.
  ``scripts/check_found_on_main_spurious_rate.py`` is the worked exemplar of
  that remedy: it catches the refusal in ``main()`` and returns a reserved
  exit 3, off the exit-1 rung its own contract already spends on "gating
  offenders found" and "backend not configured".

The guard deliberately does NOT require an absolute path and does not try to
verify the target is "the live store". The documented, working invocation for
both escalation scripts is repo-root-relative, and run from the project root it
resolves correctly; requiring absoluteness would break a correct invocation to
catch a case the populated check already catches. Pinning an expected live path
would hardcode one deployment into a general-purpose script — the mistake
``store_mutation_preflight.py::resolve_history_dir``'s docstring already warns
about for ``~/.mem0``.

Everything above is a DATED MEASUREMENT (tasks 4319, 5468), not an invariant, and must
not be restated as one: nothing enforces the rule mechanically, so a script
added tomorrow is unguarded by default.

Pure stdlib, ``sqlite3`` included. No probe write, no mem0 import, no network,
no backend import.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from enum import Enum
from pathlib import Path

__all__ = [
    'TargetStoreMissing',
    'assert_queue_dir_populated',
    'assert_task_store_populated',
    'task_store_path',
]

_TASKS_TABLE = 'tasks'


class TargetStoreMissing(RuntimeError):
    """Raised when the store an operation would touch is missing or EMPTY.

    An empty placeholder counts as missing: it is what the auto-creates leave
    behind for a run aimed at the wrong location, not the store itself.

    CALLER CONTRACT: on this exception the caller MUST NOT begin the scan, let
    alone the mutation. Refusing before the scan is the entire point — the
    failure mode being prevented is a CONFIDENT EMPTY REPORT from a store that
    was conjured into existence by the very act of looking at it, which reads
    to an operator exactly like a clean run.

    Shares ``RuntimeError`` with
    ``store_mutation_preflight.py::StoreMutationUnavailable`` so a caller can
    catch both preflight refusals together, and follows the same convention of
    stating the caller contract here rather than at each call site. It is NOT
    that guard's failure mode: see this module's docstring.

    The remedy is never to weaken the guard. Point the operation at the store
    that actually exists — an absolute ``--queue-dir``, or the main checkout as
    ``--project-root`` — and remove any empty placeholder a mis-targeted run
    left behind.
    """


class _StoreState(Enum):
    MISSING = 'missing'
    EMPTY = 'empty'
    POPULATED = 'populated'
    UNDETERMINED = 'undetermined'


def task_store_path(project_root: Path | str) -> Path:
    """Return the tasks.db a ``--project-root`` resolves to.

    Lives here, rather than being spelled out at each call site, so the four
    task-store scripts in ``fused-memory/scripts/`` (``audit_duplicate_tasks``,
    ``audit_found_on_main_provenance``, ``check_found_on_main_spurious_rate``,
    ``correct_found_on_main_backlog``) all guard the SAME path -- ``scripts/``
    is not a package, so a shared module is the only place they can agree.

    Those four call :func:`assert_task_store_populated`, which applies this
    derivation for them; the function stays public for a caller that needs the
    path WITHOUT the assertion -- reporting it, or guarding a store it reaches
    by some other route.

    Mirrors
    ``fused_memory/backends/sqlite_task_backend.py::SqliteTaskBackend._db_path``
    by construction, deliberately WITHOUT importing it: that is a private
    staticmethod, and importing the backend module at guard time would defeat
    the point of refusing before the backend exists. The duplication is one
    three-segment join, and a drift between the two shows up as a guard that
    refuses a project that works (loud), never as one that passes a project
    that does not (silent). ``_TASKS_TABLE`` mirrors the backend's table name
    on the same terms.
    """
    return Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'


def _refuse_unless_populated(
    path: Path | str,
    classify: Callable[[Path], _StoreState],
    *,
    operation: str,
    what: str,
    remedy: str,
) -> None:
    """Refuse *operation* unless *classify* finds the store at *path* populated.

    The one enforcement point both arms share. *path* is resolved once and
    reported RESOLVED, because the measured trigger is a relative default
    (``./data/reconciliation/escalations``) that, echoed back verbatim, tells
    an operator nothing about which directory was consulted. *what* names the
    store in operator words; *remedy* is the corrective invocation, the part
    an operator acts on.
    """
    resolved = Path(path).expanduser().resolve()
    state = classify(resolved)
    if state in (_StoreState.POPULATED, _StoreState.UNDETERMINED):
        return
    if state is _StoreState.MISSING:
        raise TargetStoreMissing(
            f'Refusing to begin {operation!r}: {what} does not exist at '
            f'{str(resolved)!r}. Proceeding would CREATE it empty and report a '
            f'clean run — both the escalation queue and tasks.db auto-create '
            f'silently, so a wrong target is indistinguishable from a quiet one. '
            f'Remedy: {remedy}'
        )
    raise TargetStoreMissing(
        f'Refusing to begin {operation!r}: {what} at {str(resolved)!r} exists '
        f'but holds nothing. That is exactly what the store a run aimed at the '
        f'wrong location auto-creates looks like, so a scan could only report '
        f'a clean run it cannot vouch for; if it is leftover from such a run, '
        f'remove it. Remedy: {remedy}'
    )


def _classify_queue_dir(queue_dir: Path) -> _StoreState:
    if not queue_dir.exists():
        return _StoreState.MISSING
    try:
        first = next(queue_dir.iterdir(), None)
    except OSError:
        return _StoreState.UNDETERMINED
    return _StoreState.EMPTY if first is None else _StoreState.POPULATED


def _classify_task_store(db: Path) -> _StoreState:
    # exists() must come first: a missing path and a directory both raise
    # OperationalError on open, so the exception cannot tell them apart.
    if not db.exists():
        return _StoreState.MISSING
    try:
        connection = sqlite3.connect(f'{db.as_uri()}?mode=ro', uri=True)
        try:
            has_table = connection.execute(
                'SELECT 1 FROM sqlite_master WHERE type = ? AND name = ?',
                ('table', _TASKS_TABLE),
            ).fetchone() is not None
            has_row = has_table and connection.execute(
                f'SELECT EXISTS (SELECT 1 FROM {_TASKS_TABLE})',
            ).fetchone()[0] == 1
        finally:
            connection.close()
    except sqlite3.Error:
        return _StoreState.UNDETERMINED
    return _StoreState.POPULATED if has_row else _StoreState.EMPTY


_TASK_STORE_WHAT = 'the project task store (tasks.db)'

_TASK_STORE_REMEDY = (
    'pass the MAIN checkout as --project-root. A task worktree has no '
    '.taskmaster/ (it is neither present in nor tracked by one), and '
    'SqliteTaskBackend.get_tasks auto-creates an empty tasks.db and returns '
    '{"tasks": []} for ANY --project-root rather than raising.'
)

_QUEUE_DIR_WHAT = 'the durable escalation queue directory'

_QUEUE_DIR_REMEDY = (
    'pass an absolute --queue-dir, or run from the project root. The default '
    '--queue-dir is the RELATIVE ./data/reconciliation/escalations, so a run '
    'from anywhere else — a task worktree in particular — targets a '
    'different, non-existent queue.'
)


def assert_task_store_populated(project_root: Path | str, *, operation: str) -> None:
    """Refuse *operation* unless ``<project_root>``'s tasks.db already holds a task.

    The entry point the four task-store scripts in ``fused-memory/scripts/``
    actually call. Only ``operation`` varies between them: the target
    derivation, the ``what`` and the ``remedy`` are constant across the whole
    family, so they live here once rather than once per script. The same SPOT
    argument :func:`task_store_path` already makes about the path applies with
    more force to the prose — a remedy copied per script is a remedy that
    drifts per script, and it is the sentence an operator actually acts on.

    Deliberately does NOT log. The refusal carries the operation, the RESOLVED
    target, why it matters and the remedy; a ``logger.error`` restating that
    beside the traceback puts the same paragraph on stderr twice, and the
    per-script ones it replaces interpolated the UNRESOLVED path — the exact
    verbatim echo the refusal resolves the path to avoid. A caller wanting a
    run-log record should catch and log the exception, whose message is
    already the better line.
    """
    _refuse_unless_populated(
        task_store_path(project_root),
        _classify_task_store,
        operation=operation,
        what=_TASK_STORE_WHAT,
        remedy=_TASK_STORE_REMEDY,
    )


def assert_queue_dir_populated(queue_dir: Path | str, *, operation: str) -> None:
    """Refuse *operation* unless the escalation queue directory holds an entry.

    The queue-side twin of :func:`assert_task_store_populated`, and the entry
    point the escalation scripts call. See that docstring for why the
    family-constant text and the absence of logging live here rather than at
    each call site.
    """
    _refuse_unless_populated(
        queue_dir,
        _classify_queue_dir,
        operation=operation,
        what=_QUEUE_DIR_WHAT,
        remedy=_QUEUE_DIR_REMEDY,
    )
