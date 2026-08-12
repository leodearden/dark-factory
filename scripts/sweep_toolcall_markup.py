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

import os
import sys
from pathlib import Path
from typing import Any, NamedTuple

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
    'NEVER_TOUCH',
    'PREFILTER_NEEDLES',
    'REASON_NEVER_TOUCH',
    'REASON_UNSANCTIONED_PLAN_LOCATION',
    'Refusal',
    'Repair',
    'ResolvedTarget',
    'Target',
    'detect',
    'discover_targets',
    'repair',
    'resolve_write_target',
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


# ---------------------------------------------------------------------------
# The must-not-touch guard, and write-target resolution.
# ---------------------------------------------------------------------------

#: Repo-relative paths this sweep must NEVER rewrite, whatever it is handed.
#:
#: Both are COMMITTED EVIDENCE that legitimately quotes leak specimens, so
#: their corrupted-looking strings are the artifact, not a defect:
#:
#: * ``docs/task-recovery-2026-05-13/worktree-inventory.json`` — 355 KB of
#:   git-tracked recovery inventory quoting real specimens. It is replicated
#:   into EVERY worktree checkout, so it appears beneath every orphaned lane;
#:   that is what makes it the likeliest thing a widened plans glob would eat.
#: * ``docs/toolcall-xml-leak-sweep-2026-08-05/dry-run-report.json`` — the 41
#:   verbatim leak records captured by the earlier dry-run sweep. "Repairing"
#:   the report of a leak destroys the record OF the leak.
#:
#: Finding 4 of the capability manifest names both. This constant is strictly
#: redundant against today's :func:`discover_targets`, which cannot yield
#: either — and that redundancy is the point. It is defence in depth against a
#: future widening of the globs, and it makes the rule greppable rather than
#: implicit in a path pattern.
NEVER_TOUCH: frozenset[str] = frozenset({
    'docs/task-recovery-2026-05-13/worktree-inventory.json',
    'docs/toolcall-xml-leak-sweep-2026-08-05/dry-run-report.json',
})

#: Refusal reasons. Named constants rather than inline strings so the report
#: renderer, the tests and the operator all key on ONE spelling.
REASON_NEVER_TOUCH = 'never-touch'
REASON_UNSANCTIONED_PLAN_LOCATION = 'unsanctioned-plan-location'

#: The shared meta-root a lane's plan.json is symlinked into.
_META_ROOT = ('.worktrees', '.task-meta')


class ResolvedTarget(NamedTuple):
    """A target cleared for writing, with the path the swap will land on."""

    #: The path as discovered — the link, when the target is a symlink.
    path: Path
    lane: str
    #: The ``os.path.realpath``-resolved file. THIS is what ``os.replace``
    #: must land on. Landing on the LINK instead would replace it with a
    #: regular file and re-fork the lane and meta-root copies — the esc-5205-9
    #: stale-plan divergence ``plan_tools._atomic_write_plan`` documents at
    #: line 715.
    write_path: Path


class Refusal(NamedTuple):
    """A target this sweep declines to write, and why.

    Returned as a VALUE, never raised-and-swallowed. ``scripts`` is inside
    ``shared/tests/silent_fallthrough_scan.py``'s ``_SCOPE_ROOTS``, so a broad
    ``except Exception`` funnelling into a default would trip the ratchet — and
    would also discard the reason a human needs in order to adjudicate the
    refusal. Every refusal reaches the report.
    """

    path: Path
    lane: str
    reason: str


def _matches_never_touch(path: Path) -> bool:
    """True if *path* ends with a :data:`NEVER_TOUCH` entry, component-wise.

    Anchored on a ``/`` boundary so a sibling like
    ``…/not-the-worktree-inventory.json`` cannot match by mere suffix.
    """
    text = path.as_posix()
    return any(text == rel or text.endswith('/' + rel) for rel in NEVER_TOUCH)


def _is_sanctioned_plan_location(resolved: Path, root_real: Path) -> bool:
    """True if *resolved* is one of the two shapes a plan may legally be.

    Measured at plan time: all five live ``.worktrees-orphaned/*/.task/
    plan.json`` are ABSOLUTE symlinks into ``<root>/.worktrees/.task-meta/
    <id>/plan.json``. Both that shape and a plain file in the orphaned
    worktree's own ``.task/`` are sanctioned; anything else means the link is
    not what this sweep believes it is, and following it would write through to
    an unknown file.
    """
    try:
        parts = resolved.relative_to(root_real).parts
    except ValueError:
        return False  # resolves outside the repo root entirely
    if len(parts) != 4:
        return False
    if parts[0] == _ORPHANED_DIR and parts[2:] == _PLAN_TAIL:
        return True
    return parts[:2] == _META_ROOT and parts[3] == 'plan.json'


def resolve_write_target(target: Target, root: Path | str) -> ResolvedTarget | Refusal:
    """Clear *target* for writing, or refuse it with a reason.

    Resolves symlinks FIRST and matches the guards against the resolved path as
    well as the literal one. Checking only the discovered path would be
    defeated by exactly the shape this corpus is full of — every live orphaned
    plan is a symlink — so a link pointing at committed evidence has to be
    caught by its target, not its name.

    :data:`NEVER_TOUCH` is checked BEFORE the plan-location gate so the refusal
    names the real hazard rather than a generic location miss.

    This function performs no I/O beyond path resolution and never writes.
    """
    root_real = Path(os.path.realpath(root))
    resolved = Path(os.path.realpath(target.path))

    if _matches_never_touch(target.path) or _matches_never_touch(resolved):
        return Refusal(path=target.path, lane=target.lane, reason=REASON_NEVER_TOUCH)

    if target.lane == LANE_PLANS and not _is_sanctioned_plan_location(
        resolved, root_real
    ):
        return Refusal(
            path=target.path,
            lane=target.lane,
            reason=REASON_UNSANCTIONED_PLAN_LOCATION,
        )

    return ResolvedTarget(path=target.path, lane=target.lane, write_path=resolved)


# ---------------------------------------------------------------------------
# repair_document — the uniform per-OBJECT rule.
# ---------------------------------------------------------------------------
#
# WHY A PER-OBJECT RULE RATHER THAN A TOOL SCHEMA (design decision 1).
#
# `repair()` wants (param, schema_params, supplied) — the shape of a live MCP
# tool call. A persisted JSON document has no tool call, so the mapping has to
# be invented. The sound one is per containing OBJECT:
#
#   schema_params = that object's OWN keys   — a recovered name must name a
#                                              real field of this very record;
#   legal targets = sibling keys whose current value is a string HOLE
#                   (empty or whitespace-only);
#   supplied      = every other key, so repair()'s own B8/B9 accept conditions
#                   do the refusing rather than a second policy layer here.
#
# Delegating to `orchestrator.mcp.plan_tools._repair_plan_fields` was the first
# choice (INV-5, no lockstep duplication) and it imports fine under a full
# venv — but the gated test command for this module is
# `uv run --project shared pytest scripts/tests/`, and `import fastmcp` FAILS
# in that env (re-measured this iteration), so plan_tools cannot be imported
# where its behaviour would be exercised.
#
# The uniform rule needs no tool schema at all, and it is strictly NARROWER
# than a parameter-name-keyed recovery on both hazards plan_tools names:
#   * a recovered name that is not already a key of this record is refused, so
#     no junk key (`step_type` beside `type`) can be invented;
#   * a target whose current value is not a STRING is always in `supplied`, so
#     a bare str can never land on `evidence: []` or on a `files` list and
#     silently change that field's type.
# Measured parity on the live corpora: identical 26/26 escalation repairs to
# the tool-schema table, and identical 5/5 `rationale` repairs to epsilon's own
# `_repair_plan_fields`. The envelope-literal enumeration itself is still taken
# solely from `shared.toolcall_markup`, which is the INV-5 property that
# actually matters here.

#: A repair was applied: the field was truncated to its clean prefix and every
#: recovered sibling restored.
ACTION_REPAIRED = 'repaired'
#: The string was flagged by detect() but repair() declined. The value is left
#: BYTE-IDENTICAL and the reason is reported for human adjudication.
ACTION_REFUSED = 'refused'

#: The tail parsed and every recovered name IS a field of this record, but at
#: least one of them currently holds authored content (or a non-string value)
#: rather than an empty string hole. Restoring would DISPLACE that content,
#: which design decision 2 forbids: the swallowed text still survives inside
#: the corrupted field, so refusing loses nothing while overwriting would
#: destroy a real value to rescue another.
REASON_NO_STRING_HOLE_TARGET = 'no-string-hole-target'
#: repair() declined even with the string-hole constraint lifted: the tail did
#: not parse with zero leftover, or a recovered name is not a field of this
#: record at all, or the clean prefix would itself still carry markup.
REASON_UNREPAIRABLE = 'unrepairable'
#: The tail names the very field it mis-closed. Applying it would make the
#: field's clean prefix and its recovered value fight over one key, and one of
#: them would be silently dropped — so this refuses instead.
REASON_SELF_NAMING_TAIL = 'self-naming-tail'


class Outcome(NamedTuple):
    """What happened to one detect()-flagged string, and where."""

    #: Dotted/indexed location within the document, e.g.
    #: ``design_decisions[1].rationale``. Built for the operator's report.
    json_path: str
    field: str
    action: str
    #: Names restored into their holes. EMPTY on a successful repair is the
    #: B4 last-parameter case (the mis-closed field was the call's final
    #: argument, so nothing was dropped) — every corrupted plan rationale
    #: measured live is this shape.
    recovered_names: tuple[str, ...]
    #: '' when the action is a repair.
    reason: str


def _join(path: str, key: str) -> str:
    """Extend a json path by one object key."""
    return f'{path}.{key}' if path else key


def _string_holes(node: dict) -> set[str]:
    """Keys of *node* whose current value is an empty/whitespace-only string.

    The "hole" concept is adopted from ``plan_tools._is_authored`` and
    TIGHTENED for this sweep: a legal target must additionally be STRING-typed.
    ``None`` and empty containers are holes to plan_tools but are NOT targets
    here, because filling one with a recovered ``str`` would change that
    field's type. Non-string holes are simply left in ``supplied``, which
    routes them through repair()'s existing B9 refusal instead of adding a
    second policy layer.
    """
    return {
        key
        for key, value in node.items()
        if isinstance(value, str) and not value.strip()
    }


def _classify_refusal(value: str, field: str, schema: set[str]) -> str:
    """Name the reason repair() declined *value*.

    Re-runs repair() ONCE with the string-hole constraint lifted (``supplied``
    empty). If it then succeeds, the tail parsed and every recovered name is a
    real field of this record, so the only thing that blocked the real call was
    a target already holding content — :data:`REASON_NO_STRING_HOLE_TARGET`.
    Otherwise the refusal is structural — :data:`REASON_UNREPAIRABLE`.

    Deliberately re-uses repair() rather than re-parsing the tail here: this
    module implements no matching or parsing of its own (INV-5), so the reason
    is DERIVED from the same algorithm that made the decision. It runs only on
    the refusal path, never on the clean or repaired paths.
    """
    if repair(value, field, schema, ()) is not None:
        return REASON_NO_STRING_HOLE_TARGET
    return REASON_UNREPAIRABLE


def _repair_dict(node: dict, path: str, outcomes: list[Outcome]) -> tuple[dict, bool]:
    """Repair one object's own string fields, then recurse into its children.

    Copy-on-write: *node* itself is returned unchanged unless something
    actually changed, so a clean document comes back by IDENTITY. That is not
    tidiness — the sweep decides whether to rewrite a file by whether the
    document changed, so a repairer that copied unconditionally would make
    every file look dirty and rewrite the whole corpus.
    """
    working = node
    changed = False

    for key in list(node.keys()):
        value = working[key]

        if isinstance(value, str):
            if detect(value) is None:
                continue
            # Recomputed per FIELD, not once per object: a hole filled by an
            # earlier repair in this same pass is no longer a hole, and two
            # corrupted fields must never both claim it.
            schema = set(working.keys())
            targets = _string_holes(working) - {key}
            result = repair(value, key, schema, schema - targets - {key})

            if result is None:
                outcomes.append(Outcome(
                    json_path=_join(path, key),
                    field=key,
                    action=ACTION_REFUSED,
                    recovered_names=(),
                    reason=_classify_refusal(value, key, schema),
                ))
                continue

            if key in result.recovered:
                outcomes.append(Outcome(
                    json_path=_join(path, key),
                    field=key,
                    action=ACTION_REFUSED,
                    recovered_names=(),
                    reason=REASON_SELF_NAMING_TAIL,
                ))
                continue

            if working is node:
                working = dict(node)
            working[key] = result.clean_value
            for name, recovered_value in result.recovered.items():
                working[name] = recovered_value
            changed = True
            outcomes.append(Outcome(
                json_path=_join(path, key),
                field=key,
                action=ACTION_REPAIRED,
                recovered_names=tuple(sorted(result.recovered)),
                reason='',
            ))

        elif isinstance(value, (dict, list)):
            new_child, child_changed = _repair_node(value, _join(path, key), outcomes)
            if child_changed:
                if working is node:
                    working = dict(node)
                working[key] = new_child
                changed = True

    return working, changed


def _repair_list(node: list, path: str, outcomes: list[Outcome]) -> tuple[list, bool]:
    """Recurse into a list's object/list elements, copy-on-write.

    Bare strings inside a list are deliberately NOT repaired: a list element
    has no sibling keys, so there is no object to derive schema_params from and
    nowhere legal to restore a recovered value to. Such a string would be
    truncate-only, which design decision 2 forbids.
    """
    working = node
    changed = False
    for index, item in enumerate(node):
        if not isinstance(item, (dict, list)):
            continue
        new_item, item_changed = _repair_node(item, f'{path}[{index}]', outcomes)
        if item_changed:
            if working is node:
                working = list(node)
            working[index] = new_item
            changed = True
    return working, changed


def _repair_node(node, path: str, outcomes: list[Outcome]):
    """Dispatch one node of the document walk."""
    if isinstance(node, dict):
        return _repair_dict(node, path, outcomes)
    if isinstance(node, list):
        return _repair_list(node, path, outcomes)
    return node, False


def repair_document(obj: Any) -> tuple[Any, list[Outcome]]:
    """Repair every repairable string in *obj*; return ``(new_obj, outcomes)``.

    Typed ``Any`` in and ``Any`` out on purpose: the argument is a decoded JSON
    document, so its static type is the open ``dict | list | str | int | float
    | bool | None`` union and every caller immediately subscripts the result by
    a key it knows from the record shape. Narrowing the return to ``dict``
    would be a lie for a document whose root is a list; leaving it unannotated
    made pyright infer ``dict | list`` and reject every ``result['field']`` at
    24 call sites. The real shape guarantee is structural and is asserted in
    the tests, not expressible here.

    Walks every dict in the document depth-first and applies the uniform
    per-object rule documented above. A refused string is left BYTE-IDENTICAL
    and reported with its reason; a clean document is returned by identity with
    zero outcomes.

    Restore-or-refuse (design decision 2): a repair never truncates alone. The
    swallowed text currently survives inside the corrupted field, so writing
    back only ``clean_value`` would DELETE it — destroying exactly the values
    cancelled task 3662 identified as the harm. Every applied repair is
    therefore lossless: ``clean_value`` is a prefix and every recovered value a
    verbatim substring (invariant D5, enforced by construction in repair()), so
    the union of the rewritten fields contains every byte of the original
    except the markup tags themselves.
    """
    outcomes: list[Outcome] = []
    new_obj, _changed = _repair_node(obj, '', outcomes)
    return new_obj, outcomes
