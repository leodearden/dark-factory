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

import json
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
#: The orphaned plan's symlink target no longer exists. Refused rather than
#: followed, because os.replace onto a dangling link CREATES a regular file —
#: inventing a plan for a lane whose meta-root was already reclaimed.
REASON_DANGLING_SYMLINK = 'dangling-symlink'
#: The plan resolves into a meta-root that a LIVE `.worktrees/<id>` lane still
#: shares, so rewriting it would rewrite a plan a running task is reading.
#: PRD D4 assigns those to task 3692's lazy write-back, not to this sweep;
#: this is how that split is enforced mechanically instead of by assumption.
#: Measured at plan time: `.worktrees/3415` is live, which leaves exactly one
#: repairable orphaned plan (3162) in the corpus today.
REASON_LIVE_LANE_PRESENT = 'live-lane-present'

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

    if target.lane == LANE_PLANS:
        if not _is_sanctioned_plan_location(resolved, root_real):
            return Refusal(
                path=target.path,
                lane=target.lane,
                reason=REASON_UNSANCTIONED_PLAN_LOCATION,
            )

        if not resolved.exists():
            # A dangling link. Refusing here is load-bearing: an unguarded
            # os.replace onto this path would CREATE a regular file, inventing
            # a plan for a lane whose meta-root was already reclaimed.
            return Refusal(
                path=target.path, lane=target.lane, reason=REASON_DANGLING_SYMLINK
            )

        lane_id = _meta_root_lane_id(resolved, root_real)
        if lane_id is not None and (root_real / '.worktrees' / lane_id).exists():
            return Refusal(
                path=target.path, lane=target.lane, reason=REASON_LIVE_LANE_PRESENT
            )

    return ResolvedTarget(path=target.path, lane=target.lane, write_path=resolved)


def _meta_root_lane_id(resolved: Path, root_real: Path) -> str | None:
    """The lane id *resolved* belongs to, if it lives in the shared meta-root.

    ``None`` for a plain-file plan in an orphaned worktree's own ``.task/``:
    that file is not shared with anything, so there is no lane whose liveness
    could make it unsafe. Deriving an "id" from whatever directory happened to
    be the parent would yield ``.task`` there and check a path that is not a
    lane at all.
    """
    try:
        parts = resolved.relative_to(root_real).parts
    except ValueError:
        return None
    if len(parts) == 4 and parts[:2] == _META_ROOT:
        return parts[2]
    return None


def dedupe_by_realpath(targets: list[Target]) -> list[Target]:
    """Drop targets whose realpath a previous target already claims.

    Order-preserving, so the FIRST discovery of a shared file wins and the
    result stays as deterministic as :func:`discover_targets`.

    Several orphaned lanes can symlink into the same meta-root file. Writing it
    twice in one run is not merely wasteful: the second pass would re-read the
    first pass's output, and any asymmetry between the two would surface as
    churn an operator cannot account for.
    """
    seen: set[Path] = set()
    kept: list[Target] = []
    for target in targets:
        resolved = Path(os.path.realpath(target.path))
        if resolved in seen:
            continue
        seen.add(resolved)
        kept.append(target)
    return kept


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
#: The repair walk was still applying changes when the round bound ran out.
#: Reported loudly rather than silently truncated into a plausible-looking
#: clean result — a document that stops converging is a signal that repair()'s
#: behaviour has changed, not something to paper over.
ACTION_DID_NOT_CONVERGE = 'did-not-converge'

#: How many times the walk may repeat before giving up. Small on purpose: a
#: recovered value can carry markup at most one level deep before B5 refuses
#: the parse, so anything past two rounds already means repair() is behaving
#: differently than measured. Four leaves headroom without turning a runaway
#: into a hang.
_MAX_REPAIR_ROUNDS = 4

#: Paired with :data:`ACTION_DID_NOT_CONVERGE`.
REASON_ROUND_BOUND_EXCEEDED = 'round-bound-exceeded'

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

    The walk repeats until a pass changes nothing, bounded by
    :data:`_MAX_REPAIR_ROUNDS`. The loop exists because a recovered value is a
    verbatim substring of the corrupted tail and can therefore itself carry
    markup — the nested double-leak class cancelled task 3654 identified.
    Landing such a value in a hole and stopping would leave a repairable string
    behind and break the binding "a second run reports 0" invariant.

    Measured today, the loop never runs twice: ``_parse_tail`` refuses outright
    when a recovered item contains a further mis-close (B5), so on every shape
    repair() accepts, one pass already converges and a second yields zero
    repairs on both live corpora. The loop is therefore INSURANCE — it makes
    the second-run-zero invariant STRUCTURAL rather than an empirical
    observation that a future widening of repair() could quietly invalidate.

    Exceeding the bound is reported LOUDLY as a
    :data:`ACTION_DID_NOT_CONVERGE` outcome naming the path that was still
    changing. The bound truncates the LOOP, never a repair: every repair
    already applied stays applied and the document stays valid, because a
    half-converged document is still strictly better than a corrupted one —
    and silently returning it as if it were clean is the failure mode that
    would matter.
    """  # noqa: D205
    outcomes: list[Outcome] = []
    current = obj

    for _round in range(_MAX_REPAIR_ROUNDS):
        round_outcomes: list[Outcome] = []
        current, changed = _repair_node(current, '', round_outcomes)
        outcomes.extend(round_outcomes)
        if not changed:
            return current, outcomes
        last_changed = round_outcomes
    else:
        # The bound was reached with the last pass STILL changing something.
        stalled = next(
            (o for o in last_changed if o.action == ACTION_REPAIRED), None
        )
        outcomes.append(Outcome(
            json_path=stalled.json_path if stalled is not None else '',
            field=stalled.field if stalled is not None else '',
            action=ACTION_DID_NOT_CONVERGE,
            recovered_names=(),
            reason=REASON_ROUND_BOUND_EXCEEDED,
        ))

    return current, outcomes


# ---------------------------------------------------------------------------
# Loading and gating one target.
# ---------------------------------------------------------------------------

#: The escalation statuses this sweep will rewrite. Design decision 3.
#:
#: `data/escalations` is written CONTINUOUSLY by the live queue, and this
#: script's temp-verify-replace prevents a torn READ but not a lost UPDATE: the
#: queue's own concurrency control is a `{escalation_id}.json.lock` sidecar
#: (queue.py:36-60) that this script deliberately does not take. Restricting to
#: terminal records is what makes that safe, and it costs almost nothing —
#: measured at plan time, 59 of the 60 corrupted files are already archived and
#: terminal, so exactly ONE pending record is skipped.
TERMINAL_STATUSES = frozenset({'resolved', 'dismissed'})

#: The record's status is not in :data:`TERMINAL_STATUSES` — including an
#: unrecognised one, which skips in the FAIL-SAFE direction rather than
#: guessing at a lifecycle state this sweep does not model.
REASON_NON_TERMINAL = 'non-terminal'
#: A JSON document under ``data/escalations`` that is not a record at all (no
#: ``id`` / ``status``). Excluded on SHAPE, never on "we found no markup" —
#: ``b3-state.json`` carries flagged strings and matches the glob.
REASON_NOT_AN_ESCALATION_RECORD = 'not-an-escalation-record'
#: The file could not be read or parsed. Reported so the sweep continues to the
#: next file: aborting the run on one bad file would leave the corpus
#: half-swept with no record of where it stopped.
REASON_UNPARSEABLE = 'unparseable'


class LoadedDocument(NamedTuple):
    """A target that passed its lane's gate, with its bytes and its parse."""

    target: Target
    #: The source text EXACTLY as read. Carried rather than re-read later
    #: because re-reading would race the live queue writer — the round-trip
    #: precondition must check the same bytes the parse came from.
    raw: str
    obj: Any


def _is_escalation_record(obj: Any) -> bool:
    """True if *obj* has the shape of an escalation record."""
    return isinstance(obj, dict) and 'id' in obj and 'status' in obj


def load_target(target: Target) -> LoadedDocument | Refusal:
    """Read, parse and gate one target.

    Error handling is NARROW by construction — only ``OSError``,
    ``UnicodeDecodeError`` and ``json.JSONDecodeError`` are caught, each
    recorded with a reason and returned as a value. ``scripts`` is inside
    ``shared/tests/silent_fallthrough_scan.py``'s ``_SCOPE_ROOTS``, so a broad
    ``except Exception`` funnelling into a default would trip the ratchet; more
    to the point, it would swallow a genuine bug in the repairer as though it
    were a malformed file.

    The escalation lane is gated on terminal status; the plans lane is NOT. A
    plan has no ``status`` field to read — its liveness gate is the
    ``.worktrees/<id>`` check in :func:`resolve_write_target`'s caller. Applying
    the escalation gate to plans would skip every plan in the corpus while
    still reporting a clean run.
    """
    try:
        raw = target.path.read_text(encoding='utf-8')
        obj = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return Refusal(path=target.path, lane=target.lane, reason=REASON_UNPARSEABLE)

    if target.lane == LANE_ESCALATIONS:
        if not _is_escalation_record(obj):
            return Refusal(
                path=target.path,
                lane=target.lane,
                reason=REASON_NOT_AN_ESCALATION_RECORD,
            )
        if obj.get('status') not in TERMINAL_STATUSES:
            return Refusal(
                path=target.path, lane=target.lane, reason=REASON_NON_TERMINAL
            )

    return LoadedDocument(target=target, raw=raw, obj=obj)
