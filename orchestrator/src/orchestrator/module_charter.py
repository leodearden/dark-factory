"""Sole orchestrator-side composition of Lock-charter Contract 1 for the
scheduler/harness module-cache path.

Contract 1: ``metadata.files`` is ALWAYS file-level; coarsening to depth-N
module lock keys happens at READ time only. This module is the single place
that composes the ``shared.locking`` primitives (``directory_locks``,
``strip_directory_locks``, ``files_to_modules``) into the two operations
every ``scheduler.py`` / ``harness.py`` call site needs:

- ``derive_modules``: files -> depth-coarsened module lock keys (READ path).
- ``sanitize_files_for_persist``: files -> file-level-only files (WRITE path).

``scheduler.py`` (``_get_modules``, ``seed_modules``, the blast-radius cache
write, ``_persist_files_metadata``) and ``harness.py``
(``_tag_task_modules``) route through this module instead of
re-implementing the strip -> coarsen pipeline inline (the pre-task-2122
state had 4+ divergent copies of this logic across those two files).

``workflow.py``'s plan-scoped / train-scoped module lock derivations
(``_union_train_scope``, ``_reconcile_scope_locks``, and the plan-file
blast-radius re-derivation sites in ``_plan``, ``_apply_revalidation_skip``,
and the SIMPLE_TASK lever-C path) were migrated to route through
``derive_modules`` in task 2373, closing the gap task 2122 left open (its
locked module scope was scheduler.py/harness.py only).  The ``_resume``
scope-grant conflict diagnostic (which builds the ``block_detail``'s
``additional`` module set) was likewise routed through ``derive_modules``
(task 2373 amendment) so it is computed on the same α-stripped basis as the
real conflict detection in ``_reconcile_scope_locks``.  The only remaining
raw ``files_to_modules`` uses in ``workflow.py`` are ``_check_scope_invariant``'s
task-2505 divergence tripwire, whose plan side is already α-stripped via
``sanitize_files_for_persist`` and whose metadata side reads Contract-1
file-level ``metadata.files`` — deliberately left as-is.
"""

from __future__ import annotations

import logging

from shared.locking import directory_locks, files_to_modules, strip_directory_locks

logger = logging.getLogger(__name__)


def derive_modules(files: list, depth: int, *, task_id: str = '') -> list[str]:
    """Derive depth-coarsened module lock keys from *files*.

    α strip: remove entries the α predicate (``shared.locking.is_file_path``)
    classifies as DIRECTORIES before lock derivation.  A directory entry would
    produce a subtree-wide prefix lock that blocks every task touching any file
    under that subtree (reify-3468).  Strip them so only real file siblings
    derive locks.  When ALL entries are directories the stripped list is empty
    -> ``files_to_modules`` returns ``[]`` -> callers fall through to their own
    fallback (e.g. the task-<id> synthetic lock in ``Scheduler._get_modules``).

    "No recognised file extension" is NOT the criterion, and has not been since
    dark_factory #3248: an extension-less entry whose final segment is a
    recognised name in ``shared.locking.EXTENSIONLESS_FILENAMES`` — the git
    hooks, ``LICENSE``, ``Dockerfile`` — is a real tracked FILE and is RETAINED.
    Previously every one of them was stripped, so a task declaring only such
    paths derived an empty charter and silently took the synthetic fallback,
    holding no lock on the file it was editing.  The criterion is now exactly
    what the predicate says, which is why this docstring defers to it rather
    than restating it.

    Emits one INFO diagnostic naming the stripped directory entries when any
    are found, so a run's logs can be grepped for pathologically wide
    charters.
    """
    dirs = directory_locks(files)
    if dirs:
        logger.info(
            'Task %s: α strip — rejected directory charter entries: %s',
            task_id,
            dirs,
        )
    return files_to_modules(strip_directory_locks(files), depth)


def sanitize_files_for_persist(files: list) -> list[str]:
    """Return only the file-level entries from *files*, for persisting to
    ``metadata.files``.

    Every ``metadata.files`` write path must call this before persisting so
    the field always holds genuine file-level paths (never a directory-shaped
    entry) — the incident class fixed by commit 54ec90fefc, where a
    directory entry written to ``metadata.files`` re-derived a subtree-wide
    lock on the next read.
    """
    return strip_directory_locks(files)
