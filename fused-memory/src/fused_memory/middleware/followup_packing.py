"""Candidate packs of held follow-ups: plans/followup-packing-sweep-prd.md §4.1, decisions 5-8 and 13.

Given the held pool as :class:`PackMember` rows, propose small packs that could each become one
coherent task. The module is pure (no I/O, no LLM) and imports nothing from the interceptor or the
curator, so the riders PRD can cite its reservation without pulling the curator in. The packing
sweep's tick is its only caller.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Final, Literal, get_args

from shared.declared_scope import DeclaredScopeBand
from shared.followup_items import MAX_FOLLOWUP_ITEMS
from shared.locking import files_to_modules, strip_directory_locks

__all__ = [
    'HUB_SHARE',
    'PACK_MAX_ITEMS',
    'PACK_MAX_UNION_KEYS',
    'RIDER_RESERVE_DAYS',
    'RECIPES',
    'RIDER_RESERVE_MAX_DECLINES',
    'MemberKind',
    'PackExclusion',
    'PackGenerator',
    'PackMember',
    'PackRecipe',
    'exclusion',
    'hub_keys',
    'member_text_fp',
    'pack_lock_keys',
    'rider_reserved',
    'survivor_of',
    'title_recipe',
]

PACK_MAX_ITEMS: Final = MAX_FOLLOWUP_ITEMS
PACK_MAX_UNION_KEYS = 15
HUB_SHARE = 0.02
RIDER_RESERVE_DAYS = 14
RIDER_RESERVE_MAX_DECLINES = 3

PackGenerator = Literal['parent', 'recipe']
MemberKind = Literal['followup', 'review_suggestion']

_HELD_STATUS = 'pending'
_TEXT_FP_SEPARATOR = '\x00'
_UNKNOWN_AGE = datetime.min.replace(tzinfo=UTC)


class PackExclusion(StrEnum):
    """Why a held member may not be packed. Declaration order is the precedence: first match wins.

    ``NOT_HELD`` means not in the held pool: no live hold, or not pending.
    """

    HOLDER = 'holder'
    NOT_HELD = 'not_held'
    BROAD = 'broad'
    LIVE_DEPENDENTS = 'live_dependents'
    CLAIMED = 'claimed'
    HAS_BRANCH = 'has_branch'
    HAS_PLAN = 'has_plan'
    DEDUPE_SKIPPED = 'dedupe_skipped'
    RIDER = 'rider'
    RIDER_RESERVED = 'rider_reserved'


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_aware(moment: datetime) -> bool:
    return moment.tzinfo is not None and moment.utcoffset() is not None


def _require_aware_now(now: datetime) -> None:
    _require(_is_aware(now), f'now must be timezone-aware; got {now}')


@dataclass(frozen=True)
class PackMember:
    """One held task as the generator sees it, built by the sweep from the task row.

    ``pending_since`` None means the age is unknown, read as oldest. ``rider_declines`` and
    ``rider_last_decline_reason`` follow the ``rider_ledger`` read contract of decision 13.
    """

    task_id: int
    title: str
    kind: MemberKind
    parent: int | None
    lock_keys: tuple[str, ...]
    item_count: int
    item_keys: tuple[str, ...]
    band: DeclaredScopeBand | None
    pending_since: datetime | None
    text_fp: str
    status: str
    dependencies: frozenset[int]
    held: bool
    is_holder: bool
    has_live_dependents: bool
    claimed: bool
    has_branch: bool
    has_plan: bool
    dedupe_skipped: bool
    rider_of: int | None
    rider_declines: int
    rider_last_decline_reason: str | None

    def __post_init__(self) -> None:
        _require(
            self.pending_since is None or _is_aware(self.pending_since),
            f'pending_since must be timezone-aware or None; got {self.pending_since}',
        )
        _require(
            self.kind in get_args(MemberKind),
            f'kind must be one of {get_args(MemberKind)}; got {self.kind!r}',
        )
        _require(
            self.band is None or self.band in get_args(DeclaredScopeBand),
            f'band must be None or one of {get_args(DeclaredScopeBand)}; got {self.band!r}',
        )
        _require(self.item_count >= 1, f'item_count must be at least 1; got {self.item_count}')
        _require(
            self.lock_keys == tuple(sorted(set(self.lock_keys))),
            f'lock_keys must be sorted and unique; got {self.lock_keys!r}',
        )


def pack_lock_keys(files: Iterable[str], *, lock_depth: int) -> tuple[str, ...]:
    """A member's lock keys: the lock table's derivation, directory charters dropped (decision 5)."""
    return tuple(files_to_modules(strip_directory_locks(list(files)), lock_depth))


def _sha256_16(material: str) -> str:
    # The 16-char shape of orchestrator/src/orchestrator/agents/triage.py::sha256_16, mirrored as
    # candidate_key.py mirrors it.
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def member_text_fp(title: str, description: str, details: str) -> str:
    """The fingerprint of a member's text: a pack is re-judged only when some member's changes."""
    return _sha256_16(_TEXT_FP_SEPARATOR.join((title, description, details)))


def rider_reserved(m: PackMember, *, now: datetime) -> bool:
    """Whether *m* is held back for a rider host: young, with files, declined too few times."""
    _require_aware_now(now)
    return (
        bool(m.lock_keys)
        and m.pending_since is not None
        and now - m.pending_since < timedelta(days=RIDER_RESERVE_DAYS)
        and m.rider_declines < RIDER_RESERVE_MAX_DECLINES
    )


_ExclusionRule = Callable[[PackMember, datetime], bool]

_EXCLUSION_RULES: tuple[tuple[PackExclusion, _ExclusionRule], ...] = (
    (PackExclusion.HOLDER, lambda m, _now: m.is_holder),
    (PackExclusion.NOT_HELD, lambda m, _now: not m.held or m.status != _HELD_STATUS),
    (PackExclusion.BROAD, lambda m, _now: m.band == 'broad'),
    (PackExclusion.LIVE_DEPENDENTS, lambda m, _now: m.has_live_dependents),
    (PackExclusion.CLAIMED, lambda m, _now: m.claimed),
    (PackExclusion.HAS_BRANCH, lambda m, _now: m.has_branch),
    (PackExclusion.HAS_PLAN, lambda m, _now: m.has_plan),
    (PackExclusion.DEDUPE_SKIPPED, lambda m, _now: m.dedupe_skipped),
    (PackExclusion.RIDER, lambda m, _now: m.rider_of is not None),
    (PackExclusion.RIDER_RESERVED, lambda m, now: rider_reserved(m, now=now)),
)


def exclusion(m: PackMember, *, now: datetime) -> PackExclusion | None:
    """The first exclusion that applies to *m*, or None when it may be packed."""
    _require_aware_now(now)
    return next((reason for reason, applies in _EXCLUSION_RULES if applies(m, now)), None)


def hub_keys(members: Sequence[PackMember], *, share: float = HUB_SHARE) -> frozenset[str]:
    """Lock keys declared by more than *share* of *members*, the whole held pool (decision 5)."""
    _require(0 < share < 1, f'share must be strictly between 0 and 1; got {share}')
    declared = Counter(key for m in members for key in m.lock_keys)
    return frozenset(key for key, count in declared.items() if count > share * len(members))


class PackRecipe(StrEnum):
    """A mechanical change that members of different parents can share as one task."""

    H14_RELOCATE = 'h14_relocate'
    CITE_DOC_HYGIENE = 'cite_doc_hygiene'
    SPOT_DEDUPE = 'spot_dedupe'
    TEST_SEAM = 'test_seam'
    DIAG_CODE = 'diag_code'


# v1, the table the 2026-11-04 re-run used: task 8451's details, plans/followup-packing-sweep-prd.md
# §9 item 2. A None recipe blocks: a title whose first match is that row has no recipe.
RECIPES: tuple[tuple[PackRecipe | None, re.Pattern[str]], ...] = tuple(
    (recipe, re.compile(pattern, re.IGNORECASE))
    for recipe, pattern in (
        (PackRecipe.CITE_DOC_HYGIENE, r'\bstale\b.{0,60}\b(prose|comments?|docstrings?|docs?|cites?|citations?|references?|claims?|entry|wording|narrative|framing|breadcrumbs?)\b|\b(re-?point|retarget|correct|fix|refresh)\b.{0,50}\bstale\b|line[- ]number (doc )?cites|intra-doc|rustdoc|\bdocs? sweep\b'),
        (PackRecipe.H14_RELOCATE, r'heuristic[- ]?14|inline test mod|in-module .{0,40}tests|\(\d[\d.,]*k? lines|\d[\d.,]*k? lines\)|into (its|their) own (module|file|suite|guard)|(partition|split) \S+\.(py|rs|sh|ts)\b'),
        (None, r'\bstale\b|\b(re-?point|retarget)\b|\bcit(e|es|ation|ations)\b|\bdrift(ed)?\b|intra-doc|rustdoc|-D warnings|line pin|\bdocs?\b.*\b(align|sweep|refresh)'),
        (PackRecipe.SPOT_DEDUPE, r'\bSPOT\b|heuristic 11|single[- ]source|\bdeduplicate|\bduplicat(e|ed|ion)\b|\bcopies\b|\bone (shared|owner|typed|home)\b|\bspelled twice\b'),
        (PackRecipe.TEST_SEAM, r'\bseam\b|dotted-path|stop patching|patch(es|ing)? _\w|reach(es|ing)? (into )?(the )?(module|private|internal)'),
        (PackRecipe.DIAG_CODE, r'DiagnosticCode|diag(nostic)?[_ ]code|text[- ]match'),
    )
)


def title_recipe(title: str) -> PackRecipe | None:
    """The recipe of the first :data:`RECIPES` row whose pattern occurs in *title*."""
    return next((recipe for recipe, pattern in RECIPES if pattern.search(title)), None)


def _age_key(m: PackMember) -> tuple[bool, datetime, int]:
    return (m.pending_since is not None, m.pending_since or _UNKNOWN_AGE, m.task_id)


def survivor_of(members: Sequence[PackMember]) -> int:
    """The member a pack is rewritten into: the oldest, then the lowest id; unknown age is oldest."""
    _require(bool(members), 'survivor_of needs at least one member')
    return min(members, key=_age_key).task_id
