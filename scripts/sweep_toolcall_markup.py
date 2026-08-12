"""Retro-sweep leaked tool-call markup out of TERMINAL persisted state.

Task 3691, PRD ``plans/toolcall-markup-containment-prd.md`` contract C3.

## What this sweeps — two pinned path sets, and nothing else

* ``data/escalations/**/*.json`` — escalation records, recursively (59 of the
  60 corrupted records measured live sit under ``archive/<date>/``). Only
  records in a TERMINAL status are rewritten; see :data:`TERMINAL_STATUSES`.
* ``.worktrees-orphaned/*/.task/plan.json`` — the plan artifacts of reclaimed
  worktree lanes. The exact ``.task/plan.json`` tail, never ``**/*.json``.

Discovery is an ALLOWLIST of those two shapes rather than a repo-wide ``.json``
walk, because the dominant hazard here is over-reach: an orphaned worktree is a
full checkout, so a ``**/*.json`` walk beneath one would find committed
evidence that legitimately QUOTES leak specimens and "repair" it. See
:data:`NEVER_TOUCH`.

## What it does NOT do

It never repairs LIVE state. PRD D4 splits the corpus: terminal records and
orphaned lanes are this sweep's; a live lane's plan.json belongs to task 3692's
lazy write-back at the plan-tools boundary. That split is enforced
mechanically, not by assumption — an orphaned plan whose symlink resolves into
a meta-root a LIVE ``.worktrees/<id>`` still shares is skipped and reported.

## Running it

Dry run is the DEFAULT; ``--apply`` is required to write anything::

    uv run --project shared python scripts/sweep_toolcall_markup.py
    uv run --project shared python scripts/sweep_toolcall_markup.py --apply

The ``uv run --project shared`` prefix is not optional. ``shared/__init__.py``
imports the whole package eagerly, so ``import shared.toolcall_markup`` drags
in shared's third-party dependencies even though ``toolcall_markup`` is itself
pure and stdlib-only — a dependency-free system python cannot run this script.
The same cost is recorded at ``scripts/scan_task_toolcall_leaks.py:102-112``,
which carries this identical bootstrap.

## AUTHORING HAZARD — this file spells NO envelope literal, ever

Every sentinel this module needs is imported from ``shared.toolcall_markup``
(the sole owner of the enumeration, INV-5) and never re-spelled here. That is
belt-and-braces: it keeps this from becoming a third enumeration site, AND it
keeps a raw ``chr(60)`` + ``/`` sequence out of the file text. Writing one
verbatim would force any agent editing this file to emit that literal inside
its own tool-call envelope, reproducing the very defect this script exists to
clean up — the agent's Write/Edit argument terminates early, truncating the
file and silently dropping the sibling arguments of that same call. The
rationale is recorded in full at ``shared/src/shared/toolcall_markup.py``
lines 52-62. If you need a literal here, import it; do not type it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

# shared/src bootstrap. Same idiom and same precedence argument as
# scripts/scan_task_toolcall_leaks.py:113-115 and
# scripts/repair_wiped_metadata_files.py:67-73: resolve it from __file__ and
# insert at sys.path[0], so a run inside a task worktree resolves `shared` to
# THIS checkout's copy of the envelope-literal enumeration rather than to
# whatever editable install happens to be on the path. The fused-memory/shared
# editable installs are ordinary .pth entries, so sys.path ORDER decides the
# winner and a hardcoded or install-provided path would silently test the main
# checkout's literals.
_SHARED_SRC = Path(__file__).resolve().parent.parent / 'shared' / 'src'
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))

from shared.toolcall_markup import (  # noqa: E402
    CANONICAL_OPENER_PREFIX,
    INVOKE_CLOSER,
    PREFILTER_NEEDLES,
    Repair,
    detect,
    repair,
)

__all__ = [
    'CANONICAL_OPENER_PREFIX',
    'INVOKE_CLOSER',
    'LANE_ESCALATIONS',
    'LANE_PLANS',
    'PREFILTER_NEEDLES',
    'Repair',
    'Target',
    'detect',
    'discover_targets',
    'repair',
]

# ---------------------------------------------------------------------------
# Lanes.
# ---------------------------------------------------------------------------

#: Escalation records under ``data/escalations``. Gated on terminal status.
LANE_ESCALATIONS = 'escalations'

#: Plan artifacts under ``.worktrees-orphaned/*/.task/``. NOT status-gated —
#: an orphaned lane has no status to read; its liveness gate is the
#: ``.worktrees/<id>`` check instead.
LANE_PLANS = 'plans'

#: Where the escalations lane is rooted, relative to the repo root.
_ESCALATIONS_DIR = ('data', 'escalations')

#: Where the plans lane is rooted, relative to the repo root.
_ORPHANED_DIR = '.worktrees-orphaned'

#: The exact tail an orphaned plan target must have, as path components.
_PLAN_TAIL = ('.task', 'plan.json')


class Target(NamedTuple):
    """One discovered file, tagged with the lane whose rules govern it.

    The lane travels WITH the path rather than being re-derived downstream,
    because the two lanes are gated differently (terminal-status vs
    live-lane-presence) and re-deriving the lane from the path shape at each
    gate is exactly the kind of duplicated predicate that drifts.
    """

    #: Absolute path as discovered — NOT yet realpath-resolved. Resolution is
    #: :func:`resolve_write_target`'s job and happens only on the write path.
    path: Path
    #: :data:`LANE_ESCALATIONS` or :data:`LANE_PLANS`.
    lane: str


def _has_dot_component(relative: Path) -> bool:
    """True if any component of *relative* is dot-prefixed.

    Applied to the path RELATIVE to the lane root, never to the absolute path:
    both lane roots are themselves dot-prefixed (``.worktrees-orphaned``, and
    the ``.task`` tail), and a repo checked out beneath a dotted directory
    would otherwise exclude everything.
    """
    return any(part.startswith('.') for part in relative.parts)


def discover_targets(root: Path | str) -> list[Target]:
    """Every sweepable file under *root*, sorted, deterministic.

    Returns the union of the two pinned path sets described in the module
    docstring. An absent lane directory yields nothing rather than raising:
    ``.worktrees-orphaned`` only exists once the reclaim timer has rotated at
    least one lane, so a fresh checkout legitimately has neither.

    Dot-prefixed files under ``data/escalations`` are EXCLUDED, explicitly.
    ``data/escalations/.watch-fire.json`` carries a full escalation-record
    shape but is live watcher state, so nothing about its content excludes it.
    This is the design decision 8 fork: ``glob.glob`` silently skips dotfiles
    while ``Path.rglob`` silently includes them, so the choice is made here in
    the open — and tested — instead of being inherited from whichever globbing
    API the implementation happened to reach for.

    Sorting is load-bearing, not cosmetic: an operator diffs one run's report
    against the next, and an unstable order would manufacture churn that reads
    as new corruption.
    """
    root_path = Path(root)
    targets: list[Target] = []

    escalations_dir = root_path.joinpath(*_ESCALATIONS_DIR)
    if escalations_dir.is_dir():
        for path in escalations_dir.rglob('*.json'):
            if not path.is_file():
                continue
            if _has_dot_component(path.relative_to(escalations_dir)):
                continue
            targets.append(Target(path=path, lane=LANE_ESCALATIONS))

    orphaned_dir = root_path / _ORPHANED_DIR
    if orphaned_dir.is_dir():
        for lane_dir in orphaned_dir.iterdir():
            if not lane_dir.is_dir():
                continue
            candidate = lane_dir.joinpath(*_PLAN_TAIL)
            # is_file() follows symlinks, which is what we want at DISCOVERY
            # time: all five live orphaned plans are symlinks, and a dangling
            # one is reported by the writer (with a `dangling-symlink` reason)
            # rather than silently dropped here.
            if candidate.is_file() or candidate.is_symlink():
                targets.append(Target(path=candidate, lane=LANE_PLANS))

    return sorted(targets)
