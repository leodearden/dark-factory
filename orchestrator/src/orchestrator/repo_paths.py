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
import os
from pathlib import Path

__all__ = [
    'DARK_FACTORY_ROOT_ENV',
    'rejected_dark_factory_root_override',
    'resolve_dark_factory_root',
]

logger = logging.getLogger(__name__)

#: The env var an operator sets to steer this resolver — and the var the
#: rotation's own ``cd $DARK_FACTORY_ROOT && scripts/watcher-rearm.sh`` re-arm
#: interpolates.  Named once here rather than respelled at each call site
#: (convention adopted from ``shared.reify_checkout.REIFY_ROOT_ENV``), so one
#: export cannot be half-honored by a consumer that typo'd the name.
DARK_FACTORY_ROOT_ENV = 'DARK_FACTORY_ROOT'

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


def _ascend_for_rearm_marker(start: Path | None = None) -> Path | None:
    """First ancestor of *start* that VALIDATES, or ``None``.

    *start* defaults to this module's own file, so the running orchestrator
    resolves the checkout whose code it is itself executing — the same
    ``__file__``-anchored trick as
    `orchestrator.agents.skill_prompt.load_skill_system_prompt` and
    `orchestrator.verify_runner.resolve_local_df_checkout`.

    The walk is keyed on ``_REARM_MARKER``, NOT on ``.git``, and that difference
    is the point.  ``resolve_local_df_checkout`` answers "which is the first
    ancestor containing ``.git``" — a different question, and one whose answer
    can be wrong here in both directions: it terminates at the FIRST ``.git``
    ancestor, so a dark-factory checkout nested under an outer git repo (vendored
    tree, a repo-of-repos, an editable install under a parent workspace) resolves
    to that outer repo and the walk gives up, even though a valid tooling root
    sits right above it.  Ascending for the marker itself walks PAST any ancestor
    that cannot serve a re-arm and keeps looking.  It also keeps this ~140-line
    path utility from importing the 3k-line ``verify_runner`` (and, transitively,
    the whole verify/RemoteRunner/sccache stack) for a ten-line ancestor walk —
    a layering inversion, since repo_paths is the lower-level module.

    Because every returned root has already passed `_validates`, callers may
    trust the result without re-checking the marker.
    """
    here = Path(__file__).resolve() if start is None else start
    # A file start begins the walk at its parent; a directory start is itself a
    # candidate.  Either way we ascend to the filesystem root.
    candidates = [here, *here.parents] if here.is_dir() else list(here.parents)
    for d in candidates:
        if _validates(d):
            return d
    return None


def _raw_override() -> str:
    """The ``DARK_FACTORY_ROOT`` export, stripped; ``''`` when unset or blank.

    Single-sources the "an empty/whitespace-only value counts as unset" rule so
    `resolve_dark_factory_root` and `rejected_dark_factory_root_override` cannot
    disagree about whether an operator has set the var at all.
    """
    return os.environ.get(DARK_FACTORY_ROOT_ENV, '').strip()


def rejected_dark_factory_root_override() -> str | None:
    """The raw ``DARK_FACTORY_ROOT`` export when it is set but does NOT validate.

    ``None`` when the var is unset/blank, or set to a value this module accepts.

    Exists because omitting the key from a spawned agent's ``env_overrides`` does
    NOT make it unset in the child: `shared.cli_invoke` seeds the subprocess env
    from ``os.environ`` and then applies the overrides on top, so an unresolvable
    root leaves whatever the orchestrator process itself exported INHERITED by
    the rotation.  A caller that tells the agent "DARK_FACTORY_ROOT is not set"
    in that situation is stating something false, and the agent will act on the
    stale value — if it happens to name a real directory, ``watcher-rearm.sh``'s
    ``[ -d ]`` guard passes and the agent gets a bare "No such file or directory"
    instead of the designed exit-2 diagnostic.  This helper lets the caller say
    "inherited, and known-bad" instead.

    Deliberately SILENT: `resolve_dark_factory_root` already warns about a
    rejected override, and a second warning per call would double-log the same
    misconfiguration.
    """
    raw = _raw_override()
    if not raw or _validates(Path(raw).resolve()):
        return None
    return raw


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

    Resolution order:
      1. ``DARK_FACTORY_ROOT``, when set to a path that VALIDATES.  An
         empty/whitespace-only value counts as unset (the ``export
         DARK_FACTORY_ROOT=`` shell accident) and falls through SILENTLY — that
         is the normal orchestrator-process state, not an anomaly.  The value is
         resolved to an absolute path before use, so a relative export cannot be
         reinterpreted against the spawned agent's own cwd.
      2. Otherwise `_ascend_for_rearm_marker`'s ``__file__``-anchored walk for
         the first ancestor carrying the marker.  See that function for why the
         walk is keyed on the marker rather than reusing
         `orchestrator.verify_runner.resolve_local_df_checkout`'s ``.git`` walk.
      3. Otherwise ``None``.

    A set-but-INVALID override is warned about and falls THROUGH to arm 2 rather
    than being honored or raising.  This deliberately DIVERGES from the sibling
    `shared.reify_checkout.resolve_reify_checkout` (task 3978), which honors a
    bad ``REIFY_ROOT`` VERBATIM and never falls through.  That opposite choice is
    right for ITS consumer and wrong for this one, because the two differ in what
    a bad override costs.  reify_checkout feeds a pytest GATE: honoring verbatim
    yields a loud ``pytest -rs`` skip naming the bad path, so the operator sees
    the typo and nothing silently passes.  This function feeds a SPAWNED AGENT'S
    ENVIRONMENT: honoring a stale export ships it to the rotation, whose
    ``cd $DARK_FACTORY_ROOT && scripts/watcher-rearm.sh`` then trips that
    script's own guard (``scripts/watcher-rearm.sh:150-152``, exit 2) — verbatim
    census sighting #4, i.e. re-creating the very failure this module exists to
    eliminate, and one stale export in a systemd unit would do it to every
    rotation across the fleet.  Falling through degrades to the CORRECT
    auto-resolved root while the warning keeps the misconfiguration visible in
    the orchestrator log; loud-over-silent-degradation is satisfied by the
    warning, not by propagating a value already known to be broken.

    ``None`` is a degradation, never an error: an unresolvable root must still
    let the rotation launch (with the key omitted from its environment), so the
    receiving ``watcher-rearm.sh`` guard keeps its own loud exit-2 diagnostic.
    Note that omitting the key does not UNSET it in a spawned child, which
    inherits the orchestrator's own environment — a caller that reports the
    degradation to an agent must consult `rejected_dark_factory_root_override`
    before claiming the variable is unset.
    """
    raw = _raw_override()
    if raw:
        override = Path(raw).resolve()
        if _validates(override):
            return override
        reason = (
            'not a directory'
            if not override.is_dir()
            else f'missing {"/".join(_REARM_MARKER)}'
        )
        logger.warning(
            '%s=%r rejected (%s); falling back to the auto-resolved checkout. '
            'Fix or unset the export to silence this.',
            DARK_FACTORY_ROOT_ENV,
            raw,
            reason,
        )

    root = _ascend_for_rearm_marker()
    if root is not None:
        return root

    logger.warning(
        'Could not resolve the dark-factory tooling root: %s=%s and no ancestor of '
        '%s carries %s. Spawned watcher rotations will NOT have %s overridden, so '
        'their `cd $%s && scripts/watcher-rearm.sh` re-arm cannot run.',
        DARK_FACTORY_ROOT_ENV,
        repr(raw) if raw else '<unset>',
        Path(__file__).resolve(),
        '/'.join(_REARM_MARKER),
        DARK_FACTORY_ROOT_ENV,
        DARK_FACTORY_ROOT_ENV,
    )
    return None
