"""repo_paths — resolve the dark-factory *tooling* checkout the orchestrator runs from.

Task 3605 (census 2026-08-02 §1.3; codebook ``entry-cand-20260722-3``).  A
watcher rotation dispatched for a cross-project target (e.g. reify) is told by
``skills/escalation-watcher-auto/SKILL.md`` to re-arm itself with::

    cd $DARK_FACTORY_ROOT && scripts/watcher-rearm.sh --queue-dir ... --level 1 ...

``DARK_FACTORY_ROOT`` was never set in the spawned environment, so that expanded
to ``cd  && scripts/...`` / ``/scripts/...`` and the sighted sessions guessed
(``reify/scripts/``) or swept the filesystem (``find dark-factory*``).  This
module answers the question those sessions could not: *which checkout carries
``scripts/watcher-rearm.sh``?*

The answer is deliberately NOT ``OrchestratorConfig.project_root``.  That is the
TARGET project being worked on; this is the dark-factory checkout whose code the
orchestrator process is itself executing.  For a cross-project target the two are
different repositories, and conflating them is the bug.
"""

from __future__ import annotations

import logging
from pathlib import Path

from orchestrator.verify_runner import resolve_local_df_checkout

__all__ = ['resolve_dark_factory_root']

logger = logging.getLogger(__name__)

#: Relpath, inside a candidate root, that proves the root is a dark-factory
#: checkout capable of serving a rotation's re-arm.  Deliberately the very
#: script the census sightings failed to invoke: its own guard
#: (``scripts/watcher-rearm.sh:150-152``) is what a wrong root trips with
#: exit 2, so validating against it makes a resolved root's promise honest
#: rather than aspirational.  A ``.git`` marker alone would answer only
#: "is this a git repo", which is not the question.
_REARM_MARKER = ('scripts', 'watcher-rearm.sh')


def _validates(root: Path) -> bool:
    """True iff *root* is a directory carrying the rearm-script marker."""
    return root.is_dir() and root.joinpath(*_REARM_MARKER).is_file()


def resolve_dark_factory_root() -> Path | None:
    """The dark-factory tooling checkout this orchestrator is running from.

    Returns the checkout carrying ``scripts/watcher-rearm.sh`` — the value to
    export as ``DARK_FACTORY_ROOT`` into a spawned watcher rotation — or
    ``None`` when no such checkout can be identified.

    This is the *tooling* root, explicitly distinct from
    ``OrchestratorConfig.project_root``: project_root is the TARGET repo the
    orchestrator is working on, which for a cross-project deployment (reify,
    etc.) is a different repository that contains no ``scripts/watcher-rearm.sh``
    at all.  Callers that need "where does the rearm script live" must use this,
    never project_root.

    Resolution reuses `orchestrator.verify_runner.resolve_local_df_checkout`
    for the ``__file__``-anchored ancestor walk rather than forking a second
    ascent; this function adds only the validation that ancestor actually
    carries the marker.  ``None`` is a degradation, never an error: an
    unresolvable root must still let the rotation launch (with the key omitted
    from its environment), so the receiving ``watcher-rearm.sh`` guard keeps its
    own loud exit-2 diagnostic.
    """
    root = resolve_local_df_checkout()
    if root is not None and _validates(root):
        return root
    return None
