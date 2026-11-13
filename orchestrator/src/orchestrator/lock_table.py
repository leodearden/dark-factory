"""Module locks and park reservations: which tasks may hold which modules at once.

The table holds no event store. The Scheduler owns the only instance and emits
every lock and park event from these methods' return values.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import NamedTuple

from shared.locking import modules_conflict, normalize_lock

from orchestrator.config import OrchestratorConfig
from orchestrator.id_tie_break import id_tie_break
from orchestrator.pin_reservation import (
    Blocker,
    BlockerKind,
    ParkPriority,
    PinOrder,
    ReservationSource,
    is_pin_rank,
    park_rank,
    rank_payload,
)
from orchestrator.time_sources import resolve_wall_time_source

logger = logging.getLogger(__name__)

#: The holder limit of a key no count of holders can fill: every limit test is
#: ``len(<set of task ids>) >= limit``, and no Python collection is that long.
UNLIMITED_HOLDERS = sys.maxsize


class _ParkKey(NamedTuple):
    """THE park order (task 7799): the smaller key outranks; see ``plans/scheduler-dispatch-scoring-and-lock-layer-prd.md`` §7, Amendment (task 7799)."""

    rank: int
    id_rank: float
    owner: str

    @classmethod
    def of(cls, owner: str, rank: int) -> _ParkKey:
        return cls(rank, *id_tie_break(owner))



@dataclass(frozen=True)
class UncontestedAcquisition:
    """What a partial widen took, and each key another task holds or parks."""

    acquired: tuple[str, ...]
    contested: tuple[Blocker, ...]

    def __post_init__(self) -> None:
        both = sorted(set(self.acquired) & {blocker.module for blocker in self.contested})
        if both:
            raise ValueError(f'lock keys {both} cannot be both acquired and contested')


class ModuleLockTable:
    """Hierarchical module locking — two modules conflict if one is a prefix
    of the other (parent/child), but siblings are independent.

    Examples (all conflict):
        autopilot/analyze  <->  autopilot/analyze/asr   (parent <-> child)
        src/server         <->  src/server               (exact match)

    Examples (no conflict):
        autopilot/analyze/asr  <->  autopilot/analyze/speech  (siblings)
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        wall_time_source: Callable[[], datetime] | None = None,
    ):
        self._limits: dict[str, int] = {}
        self._held: dict[str, set[str]] = {}  # task_id -> set of held modules
        # normalized_module -> LIFO stack of (owner_task_id, priority_rank).
        # Top = last element = the ACTIVE reservation.  Preemption by an entry
        # with a smaller _ParkKey PUSHES (shadows) the top entry instead of
        # destroying it.  Restoration happens when the top entry is removed.
        # INV-3: every stack is strictly ordered by _ParkKey, smallest key on
        # top (task 7799), so its depth is bounded by one entry per owner per
        # band (tier, pin, task 6040).  An owner's pin entry sits above its
        # own tier entry.
        self._parked: dict[str, list[tuple[str, int]]] = {}
        # task_id -> ISO8601 timestamp of first install_parks call for that owner
        self._park_install_at: dict[str, str] = {}
        self._config = config
        # WALL clock for park install stamps (task 3823 / PRD task η):
        # _park_install_at holds ISO-8601 strings that must be comparable with
        # the Scheduler's own _wall_now() when park_age_secs() computes an age,
        # and with the timestamps in the event log.  Resolved through the
        # shared helper so there is exactly one None-fallback rule for wall
        # clocks.
        self._wall_time_source: Callable[[], datetime] = resolve_wall_time_source(
            wall_time_source
        )

    # --- Hierarchy helpers ---

    @staticmethod
    def _conflicts(a: str, b: str) -> bool:
        """Two modules conflict if one is a prefix of the other (or exact match).

        Thin delegate to ``shared.locking.modules_conflict`` so the prefix rule
        has a single definition shared with the dashboard's holder lookup.
        """
        return modules_conflict(a, b)

    def _conflicting_holders(
        self, module: str, exclude_task: str | None = None
    ) -> set[str]:
        """The *other* tasks holding a lock that conflicts with ``module``."""
        return {
            task_id
            for task_id, task_modules in self._held.items()
            if task_id != exclude_task
            and any(self._conflicts(held, module) for held in task_modules)
        }

    def _count_conflicts(self, module: str, exclude_task: str | None = None) -> int:
        """Count how many *other* tasks hold a lock that conflicts with ``module``."""
        return len(self._conflicting_holders(module, exclude_task=exclude_task))

    # --- Park (reservation) helpers ---

    def _precedence_blocking_parks(
        self,
        module: str,
        task_id: str,
        *,
        ignore_owners: frozenset[str],
    ) -> list[tuple[str, str]]:
        """``(owner, parked_key)`` for every foreign active top *task_id* cannot pass on *module*.

        A buried (shadowed) reservation is never consulted — only the active
        top of each stack (INV-2).  Order is ``_parked`` iteration order.

        INV-8, park precedence (task 7799): a foreign conflicting top blocks a
        requester that owns parks only when that entry's :class:`_ParkKey`
        precedes the requester's best key (:meth:`_best_park_key`, over every
        entry it holds, at any level and on any key).  A requester that owns
        no park is blocked by every conflicting foreign top.  So every park
        wait points at an owner whose best key is strictly smaller than the
        waiter's, and the wait-for graph between park owners is acyclic in
        every reachable state; the owner with the smallest key waits only on
        live holders.  Rationale: ``plans/scheduler-dispatch-scoring-and-lock-layer-prd.md``
        §7, Amendment (task 7799).

        *ignore_owners* (task 3823 / PRD C7) names park owners whose parks are
        treated as absent for this request — the EASY-backfill admission
        bypass.  It is scoped to the park rule alone and has no bearing on the
        held-lock limit gate.
        """
        own = self._best_park_key(task_id)
        blocking: list[tuple[str, str]] = []
        for parked_module, stack in self._parked.items():
            if not stack:
                continue
            owner, rank = stack[-1]  # active TOP only (INV-2)
            if owner == task_id or owner in ignore_owners:
                continue
            if not self._conflicts(parked_module, module):
                continue
            if own is not None and own < _ParkKey.of(owner, rank):
                continue
            blocking.append((owner, parked_module))
        return blocking

    def _iter_blocking_park_owners(
        self,
        module: str,
        task_id: str,
        *,
        ignore_owners: frozenset[str] = frozenset(),
    ) -> Iterator[str]:
        """EVERY owner of an ACTIVE park that blocks *task_id* on *module*.

        THE single precedence-aware park-blocking rule (INV-8, over
        :meth:`_precedence_blocking_parks`).  ``_blocking_park_owner`` is the
        first-match view over it, ``_is_parked_blocks`` the bool view, and
        ``park_owners_blocking`` the naming view — one rule, three views
        (INV-5).  Forking the precedence logic is what would let admission believe it
        can borrow through a park that ``try_acquire`` then refuses: churn with
        no dispatch.

        More than one owner can block a single requested module — two owners
        parked on different children of the same requested parent both
        conflict with it — and admission needs ALL of them, because
        ``try_acquire`` re-checks every park and admits only the owners it was
        handed.

        Capacity (task 7313): a park takes one of its key's slots instead of
        reserving the key exclusively.  A park on key K that
        :meth:`_precedence_blocking_parks` names blocks when either term fires:

        (a) the hold term — a park counts as its owner's hold, so the request
            is refused once *module* is at ``_limit_for(module)`` counting the
            foreign holders and the blocking park owners, each task once;
        (b) the owner's-slot term — a park keeps one slot of its own key, so
            the request is refused when granting it would fill K for the owner
            (:meth:`_park_owner_slot_taken`).

        A limit of 1 on *module* makes the hold term always fire, and a limit
        of 1 on K makes the owner's-slot term fire for every requester not
        already counted on K, so a limit-1 park admits no new occupant,
        hierarchical cases included.  The hold term counts park owners first
        and scans holders only when the owners alone leave *module* room, so a
        limit-1 request pays no extra scan.  The rule is
        monotone in *ignore_owners*: ignoring an owner only removes hold-term
        occupants, and the owner's-slot term reads holders only, never
        *ignore_owners* — so once admission's named owners are ignored, no
        remaining park starts blocking.

        The hold term needs the full count, so nothing is yielded before every
        park has been scanned; a generator keeps every view's call shape.
        """
        parks = self._precedence_blocking_parks(module, task_id, ignore_owners=ignore_owners)
        if not parks:
            return
        park_owners = {owner for owner, _key in parks}
        limit = self._limit_for(module)
        if len(park_owners) >= limit or len(
            park_owners | self._conflicting_holders(module, exclude_task=task_id)
        ) >= limit:
            yield from (owner for owner, _key in parks)
            return
        for owner, key in parks:
            if self._park_owner_slot_taken(owner, key, task_id):
                yield owner

    def _park_owner_slot_taken(self, owner: str, key: str, requester: str) -> bool:
        """Would granting *requester* a key conflicting with *key* take *owner*'s slot on it?

        True when the grant is what leaves the park owner's own later
        acquisition of *key* no room under the held gate: *requester* does not
        already hold a key conflicting with *key*, and the holders conflicting
        with *key* plus *owner* itself already reach ``_limit_for(key)``.  A
        requester already counted on *key* — say, expanding to a sibling under
        a parked parent — leaves the owner's room unchanged.
        """
        holders = self._conflicting_holders(key)
        if requester in holders:
            return False
        return len(holders | {owner}) >= self._limit_for(key)

    def _blocking_park_owner(
        self,
        module: str,
        task_id: str,
        *,
        ignore_owners: frozenset[str] = frozenset(),
    ) -> str | None:
        """The FIRST active park owner blocking *task_id* on *module*, or None.

        The first-match view of :meth:`_iter_blocking_park_owners`.  Enough for
        the gate (``_is_parked_blocks`` only needs to know THAT something
        blocks).  Callers that must reason about every blocking owner —
        admission — drain the generator instead.
        """
        return next(
            self._iter_blocking_park_owners(
                module, task_id, ignore_owners=ignore_owners
            ),
            None,
        )

    def park_owners_blocking(self, task_id: str, modules: list[str]) -> list[str]:
        """Distinct active-top park owners that would make ``try_acquire`` refuse.

        Sorted for determinism.  Normalizes *modules* exactly as
        ``try_acquire`` does, so the answer is about the same keys the gate
        will actually test.

        This is C7's ``c.modules ∩ M ≠ ∅`` clause: the hierarchical
        ``_conflicts`` rule already IS the intersection test, so an empty list
        means there is nothing to backfill through and admission must refuse.

        UNIONS every blocking owner per module, not the first one found: two
        owners parked on different children of one requested parent both
        conflict with it, and a set missing the second would be handed to
        ``try_acquire`` as ``admitted_parks`` — which re-checks every park and
        would refuse on the owner nobody named.  That failure mode is silent
        (admission pays for the full prediction pipeline every tick and the
        acquire always fails), so it has to be excluded here rather than
        detected downstream.

        A READ — it mutates nothing and evicts nothing.
        """
        depth = self._config.lock_depth
        normalized = {normalize_lock(m, depth) for m in modules}
        owners: set[str] = set()
        for module in normalized:
            owners.update(self._iter_blocking_park_owners(module, task_id))
        return sorted(owners)

    def blocking_holders(
        self, modules: list[str], *, exclude_task: str
    ) -> dict[str, list[str]]:
        """``{requested_module: [foreign holders]}`` for modules at/over their limit.

        Only modules that are ACTUALLY blocked appear; everything free is
        omitted, so an empty mapping means the requester could take all of
        *modules* right now.  Keys are the requested modules in normalized
        form (not the holders' own keys), because the caller asked about those.

        Honours ``_limit_for`` rather than mere presence: a module with
        headroom (``max_per_module`` > current holders) is free to take, so it
        contributes no wait.  Built from the existing ``_conflicting_holders``
        / ``_limit_for`` pair — no third conflict rule.
        """
        depth = self._config.lock_depth
        result: dict[str, list[str]] = {}
        for module in {normalize_lock(m, depth) for m in modules}:
            holders = self._conflicting_holders(module, exclude_task=exclude_task)
            if holders and len(holders) >= self._limit_for(module):
                result[module] = sorted(holders)
        return result

    def blockers(self, task_id: str, modules: list[str]) -> list[Blocker]:
        """Every holder and active park that would make ``try_acquire`` refuse.

        Composed from :meth:`blocking_holders` and
        :meth:`_iter_blocking_park_owners` over the same normalized keys the
        gate tests, so it names exactly what the gate honours (INV-5): never
        a buried park, never the requester's own.  Sorted and de-duplicated;
        empty means the acquire would succeed.  A READ.
        """
        depth = self._config.lock_depth
        found: set[Blocker] = {
            Blocker(module, holder, BlockerKind.HELD)
            for module, holders in self.blocking_holders(
                modules, exclude_task=task_id
            ).items()
            for holder in holders
        }
        for module in {normalize_lock(m, depth) for m in modules}:
            found.update(
                Blocker(module, owner, BlockerKind.PARKED)
                for owner in self._iter_blocking_park_owners(module, task_id)
            )
        return sorted(found)

    def _is_parked_blocks(
        self,
        module: str,
        task_id: str,
        *,
        ignore_owners: frozenset[str] = frozenset(),
    ) -> bool:
        """Return True iff any ACTIVE (top-of-stack) park conflicts with *module*,
        is owned by a different task that task_id does not precede (INV-2,
        INV-8), and fires the two-term capacity rule of
        :meth:`_iter_blocking_park_owners` (task 7313).

        The bool view of :meth:`_blocking_park_owner`, which carries the
        precedence logic.  Kept as a delegate rather than inlined so ``try_acquire`` /
        ``try_acquire_additional`` and every existing test keep their call
        shape, and so the naming view (``park_owners_blocking``) and this
        gating view can never disagree (INV-5).
        """
        return self._blocking_park_owner(
            module, task_id, ignore_owners=ignore_owners
        ) is not None

    def _owned_ranks(self, task_id: str) -> dict[str, int]:
        """``{key: rank}`` for every stack holding *task_id* at ANY level, top or buried.

        THE single definition of "the owner already parks this key": the
        install idempotency skip, the completion remainder, ``has_parks`` and
        :meth:`_best_park_key` all read it, so they cannot disagree about a
        buried entry.  The rank is the owner's best (lowest) one on that key.
        """
        owned: dict[str, int] = {}
        for module, stack in self._parked.items():
            ranks = [rank for owner, rank in stack if owner == task_id]
            if ranks:
                owned[module] = min(ranks)
        return owned

    def _best_park_key(self, task_id: str) -> _ParkKey | None:
        """*task_id*'s smallest :class:`_ParkKey` over every entry it holds, at any level; None without parks."""
        owned = self._owned_ranks(task_id)
        return _ParkKey.of(task_id, min(owned.values())) if owned else None

    @staticmethod
    def _covers(held_rank: int | None, rank: int | None) -> bool:
        """True iff an owner entry at *held_rank* already covers a request at *rank*.

        *held_rank* None means the owner has no entry on the key.  *rank* None
        means "at any rank".  An entry covers a request at its own rank or any
        weaker one; it never covers a stronger one, which is a re-rank.
        """
        return held_rank is not None and (rank is None or held_rank <= rank)

    def has_parks(self, task_id: str) -> bool:
        """Return True if *task_id* owns any reservation at ANY stack level (INV-5).

        A pure query over :meth:`_owned_ranks`, so an owner buried beneath a
        higher-priority shadow still counts as parked.
        """
        return bool(self._owned_ranks(task_id))

    def reservation_source(
        self, owner: str, modules: Iterable[str] | None = None
    ) -> ReservationSource:
        """Why *owner*'s parks exist, derived from their ranks (task 6040).

        PIN if any of the owner's entries, top or buried, sits in the pin band;
        otherwise FAIRNESS.  *modules* (normalized keys) narrows the question
        to the owner's best entry on each of those keys — a restore exposes
        one specific entry, and an owner holding a pin reservation elsewhere
        can have a fairness entry exposed here.  An owner with no matching
        parks reads FAIRNESS, so callers that ask after a removal get the
        automatic default rather than an error.
        """
        owned = self._owned_ranks(owner)
        keys = owned.keys() if modules is None else owned.keys() & set(modules)
        if any(is_pin_rank(owned[key]) for key in keys):
            return ReservationSource.PIN
        return ReservationSource.FAIRNESS

    def unparked_modules(
        self,
        task_id: str,
        modules: list[str],
        priority: ParkPriority | None = None,
    ) -> list[str]:
        """The completion rule's remainder (task 5308): keys *task_id* has not parked.

        Normalizes *modules* exactly as :meth:`install_parks` does, drops empty
        keys, de-duplicates, and subtracts every key the owner already parks
        at ANY stack level.  Sorted.

        For a tier *priority*, or none, an entry at ANY rank counts as owned —
        a buried entry and a pin entry included — because fairness completion
        fills coverage gaps and never re-ranks.  For a :class:`PinOrder`, a
        key counts as owned only at that pin rank or better (task 6040): the
        owner's own fairness entry does not cover its pin reservation, which
        :meth:`install_parks` then stacks above it.
        """
        depth = self._config.lock_depth
        owned = self._owned_ranks(task_id)
        floor = park_rank(priority) if isinstance(priority, PinOrder) else None
        requested = {normalize_lock(m, depth) for m in modules}
        return sorted(
            m for m in requested if m and not self._covers(owned.get(m), floor)
        )

    def install_parks(
        self, task_id: str, modules: list[str], priority: ParkPriority
    ) -> tuple[list[str], list[tuple[str, list[str]]]]:
        """Install reservations on the normalized form of *modules* for *task_id*.

        Returns ``(installed, shadowed)`` where *installed* is the list of
        normalized modules NEWLY parked, or re-ranked upward, as the active
        top by this call and *shadowed* is a list of
        ``(owner_id, modules_shadowed)`` for any parks with a larger
        :class:`_ParkKey` (a lower priority, or the same rank and a later id)
        that were PUSHED beneath the new reservation (retained, not destroyed).

        *priority* is a tier name or a :class:`PinOrder`; :func:`park_rank`
        resolves either to the stack rank (task 6040).

        Idempotent per owner (task 5308): a key *task_id* already parks at
        *priority* or better, at any stack level (:meth:`_owned_ranks`), is
        skipped, and so is a second input normalizing to a key this call
        already handled.  A key it parks at a LOWER priority is a rank
        upgrade: it runs the same conflict scan at the new rank and, unless
        blocked, a new entry goes on top at the new rank and the key is
        reported in *installed*; a blocked upgrade keeps the old entry.  The
        upgrade replaces the owner's old entry in the SAME band
        (:func:`is_pin_rank`).  A pin reservation leaves the owner's tier
        entry in place beneath it (task 6040), so releasing the pin restores
        that fairness park as it was.  Either way the owner holds at most one
        entry per band per stack, never reports shadowing itself, and is
        never downgraded.

        Preemption (INV-1): if the active TOP of a conflicting stack has a
        larger :class:`_ParkKey` than the new entry (a lower priority, or the
        same rank and a later id), the new entry is PUSHED on top of the stack
        (shadowing, not destroying, the top entry).  A top with a smaller key
        BLOCKS installation for that module (INV-3, task 7799).

        Hierarchical (different-key) conflicts: victim parks are kept under
        their ORIGINAL keys (not re-keyed onto the preemptor's key).
        Precedence-aware blocking in ``_is_parked_blocks`` (INV-8) lets the
        preemptor acquire its own module without needing to move victim entries.  This preserves each
        victim's original module identity so that ALL victims are independently
        restorable after the preemptor clears (step-12 fix).
        """
        depth = self._config.lock_depth
        rank = park_rank(priority)
        key = _ParkKey.of(task_id, rank)
        installed: list[str] = []
        # Accumulate shadows: victim_owner -> list of original module keys shadowed.
        shadow_acc: dict[str, list[str]] = {}
        # Track insertion order of shadowed owners for stable output.
        shadow_order: list[str] = []
        owned = self._owned_ranks(task_id)
        for m in modules:
            normalized = normalize_lock(m, depth)
            if not normalized or self._covers(owned.get(normalized), rank):
                continue
            # Scan only the ACTIVE TOP of each conflicting stack (INV-2).
            to_shadow: list[tuple[str, str]] = []  # (parked_m_key, victim_owner)
            blocked = False
            for parked_m, stack in list(self._parked.items()):
                if not stack:
                    continue
                owner, existing_rank = stack[-1]  # active TOP (INV-2)
                if owner == task_id:
                    continue
                if not self._conflicts(parked_m, normalized):
                    continue
                # Hierarchical conflict with a different top owner.
                if _ParkKey.of(owner, existing_rank) > key:
                    # The new entry outranks TOP → shadow (push), retain beneath.
                    to_shadow.append((parked_m, owner))
                else:
                    # TOP outranks the new entry → block our install (INV-3).
                    blocked = True
                    break
            if blocked:
                continue
            # Record shadows — all victim parks stay under their ORIGINAL keys.
            # For exact-match conflicts (parked_m == normalized), the victim's
            # entry stays in its stack; the new entry will be pushed on top
            # in the install step below.  For hierarchical (different-key)
            # conflicts, the victim park is also left untouched: precedence-
            # aware blocking in _is_parked_blocks lets the preemptor acquire its own
            # module without moving victim entries (step-12 fix).
            for parked_m, victim_owner in to_shadow:
                if victim_owner not in shadow_acc:
                    shadow_acc[victim_owner] = []
                    shadow_order.append(victim_owner)
                shadow_acc[victim_owner].append(parked_m)

            # Push task_id onto the top of the (possibly freshly created)
            # stack, dropping the same-band entry a rank upgrade replaces.
            stack = [
                (owner, entry_rank)
                for owner, entry_rank in self._parked.get(normalized, [])
                if owner != task_id or is_pin_rank(entry_rank) != is_pin_rank(rank)
            ]
            stack.append((task_id, rank))
            self._parked[normalized] = stack
            owned[normalized] = rank
            installed.append(normalized)

        # Build shadowed list in first-seen victim order, original module keys sorted.
        shadowed = [
            (owner, sorted(shadow_acc[owner]))
            for owner in shadow_order
        ]
        if installed:
            # ``setdefault``, not assignment: C7's ``park_age(p)`` is the owner's
            # TOTAL wait, so a re-install must not reset the clock — a refreshed
            # stamp would make an indefinitely-starving park read as permanently
            # fresh and keep admitting backfillers through it.
            self._park_install_at.setdefault(
                task_id, self._wall_time_source().isoformat()
            )
        return installed, shadowed

    def _remove_owner(
        self, task_id: str, *, source: ReservationSource | None = None
    ) -> list[tuple[str, list[str]]]:
        """Remove *task_id* from every stack at any level and return restored pairs.

        With *source*, only the owner's entries of that source go (task 6040):
        releasing a pin reservation leaves the owner's fairness parks where
        they are.

        For each module stack whose ACTIVE TOP was removed: after removal
        the new last element (if any) becomes the active top and is reported as a
        ``(owner, sorted_modules)`` restored pair — with *source*, possibly
        the owner's own remaining entry.  Removing a buried (non-top)
        entry leaves the top unchanged and is NOT reported as a restore (INV-4).

        Drops ``_park_install_at[task_id]`` only when task_id has no remaining
        parks at any level after removal (INV-6: preserved for shadowed owners
        who are removed from buried slots but still have other parks).

        Returns ``list[(restored_owner, sorted_modules)]`` in first-seen order.
        Callers use the returned pairs to emit ``reservation_restored`` events.

        **Hierarchical victim note**: for hierarchical (different-key) shadows,
        the hierarchical victim's own stack is NEVER touched during the
        preemptor's removal — the preemptor only appears in its own normalized
        key's stack, so ``was_top`` only fires for that key.  The victim becomes
        acquirable the instant the preemptor is removed (precedence-aware blocking in
        ``_is_parked_blocks`` stops gating on the now-absent foreign top), but no
        ``reservation_restored`` pair is returned and no event is emitted.  This
        asymmetry is intentional (documented in the plan's design decisions):
        adding cross-key restore detection would require scanning all stacks for
        newly-unblocked entries on every removal.  Exact-match shadows (same
        key) always produce a restore pair; hierarchical shadows do not.
        """
        def removed(entry: tuple[str, int]) -> bool:
            owner, rank = entry
            return owner == task_id and (
                source is None or ReservationSource.source_of_rank(rank) is source
            )

        # Track which modules had a removed entry as their ACTIVE TOP.
        was_top = {m for m, stack in self._parked.items() if stack and removed(stack[-1])}

        # Rebuild stacks without the removed entries.
        new_parked: dict[str, list[tuple[str, int]]] = {}
        for m, stack in self._parked.items():
            new_stack = [entry for entry in stack if not removed(entry)]
            if new_stack:
                new_parked[m] = new_stack
        self._parked = new_parked

        # Drop install_at only when task_id is fully gone.
        if not self.has_parks(task_id):
            self._park_install_at.pop(task_id, None)

        # Identify newly-exposed tops: modules where task_id WAS the top and
        # a lower entry is now exposed.
        restored_acc: dict[str, list[str]] = {}  # new_top_owner -> [modules]
        restored_order: list[str] = []
        for m in was_top:
            new_stack = self._parked.get(m)
            if new_stack:  # there is a newly-exposed lower entry
                new_top_owner, _ = new_stack[-1]
                if new_top_owner not in restored_acc:
                    restored_acc[new_top_owner] = []
                    restored_order.append(new_top_owner)
                restored_acc[new_top_owner].append(m)

        return [
            (owner, sorted(restored_acc[owner]))
            for owner in restored_order
        ]

    def clear_parks_for(self, task_id: str) -> list[tuple[str, list[str]]]:
        """Remove every reservation owned by *task_id* and return restored pairs.

        Delegates to ``_remove_owner`` for the actual removal + restore detection.
        Returns ``[(restored_owner, sorted_modules)]`` for each module whose
        newly-exposed lower entry becomes the active top — empty list when there
        are no buried entries beneath task_id's reservations (INV-4).

        Callers (Scheduler dispatch/release paths) use the returned pairs to
        emit ``reservation_restored`` events (step-8).
        """
        return self._remove_owner(task_id)

    def prune_owners(
        self,
        predicate: Callable[[str], bool],
        *,
        source: ReservationSource | None = None,
    ) -> tuple[list[str], list[tuple[str, list[str]]]]:
        """Evict every park whose owner satisfies *predicate* at ANY stack level.

        Iterates unique owners across all levels of all stacks (deduped,
        first-seen order), calls ``predicate(owner_id)`` at most once per
        owner, and removes all entries for matching owners via ``_remove_owner``.
        With *source*, only owners holding an entry of that source are asked,
        and only those entries are removed (task 6040).

        Returns ``(evicted_owners, restored_pairs)`` where *evicted_owners* is
        the list of removed owner IDs in first-seen order and *restored_pairs*
        is a flattened list of ``(owner, sorted_modules)`` for every stack
        whose newly-exposed top is still the top once the whole sweep is
        done.  The Scheduler's ``_park_gc`` sweep loop uses restored_pairs to
        emit ``reservation_restored`` events (step-8).
        """
        # Collect unique owners in first-seen order across all stack levels.
        seen: dict[str, bool] = {}
        for stack in self._parked.values():
            for owner, rank in stack:
                if source is not None and ReservationSource.source_of_rank(rank) is not source:
                    continue
                if owner not in seen:
                    seen[owner] = predicate(owner)
        # Build evicted list.
        evicted = [owner for owner, should in seen.items() if should]
        # Remove each evicted owner and accumulate restored pairs.
        all_restored_acc: dict[str, list[str]] = {}  # deduplicate across calls
        all_restored_order: list[str] = []
        for owner in evicted:
            for restored_owner, mods in self._remove_owner(owner, source=source):
                if restored_owner not in all_restored_acc:
                    all_restored_acc[restored_owner] = list(mods)
                    all_restored_order.append(restored_owner)
                else:
                    # Merge modules for the same restored owner across multiple calls.
                    merged = sorted(set(all_restored_acc[restored_owner]) | set(mods))
                    all_restored_acc[restored_owner] = merged
        # Keep only restores that survived the rest of the sweep.  When
        # removing top Z exposes lower Y, but a later removal in this sweep
        # takes Y's entry too, the caller would otherwise emit a spurious
        # reservation_restored for an entry that is gone.  Asking the stacks
        # rather than the evicted set keeps a source-scoped sweep right: there
        # an evicted owner's other-source entry can be the surviving top.
        all_restored = [
            (owner, survivors)
            for owner in all_restored_order
            if (survivors := [m for m in all_restored_acc[owner] if self._top_owner(m) == owner])
        ]
        return evicted, all_restored

    def _top_owner(self, module: str) -> str | None:
        """The owner of *module*'s active top entry, or None for no stack."""
        stack = self._parked.get(module)
        return stack[-1][0] if stack else None

    def force_clear(
        self, owner: str
    ) -> tuple[list[str], list[tuple[str, list[str]]]]:
        """Evict all parks held by *owner* and return (modules, restored_pairs).

        Thin wrapper over :meth:`prune_owners` that pre-collects the sorted
        list of modules the owner was parked on before the eviction removes
        them from the stacks.  LIFO restoration of newly-exposed tops is
        handled by :meth:`prune_owners` → :meth:`_remove_owner`.

        Returns:
            modules: sorted list of module keys the owner was parked on.
                     Empty list if the owner has no parks (idempotent no-op).
            restored_pairs: list of (owner, sorted_modules) for each buried
                            owner newly exposed as the active top.
        """
        modules = sorted(
            m for m, stack in self._parked.items()
            if any(o == owner for o, _ in stack)
        )
        _evicted, restored = self.prune_owners(lambda o: o == owner)
        return modules, restored

    # --- Snapshot helpers (public accessors for observability) ---

    def park_age_secs(self, owner: str, *, now: datetime) -> float | None:
        """Seconds since *owner* FIRST installed a park, or ``None`` if unknown.

        **``None`` means "age unknown ⇒ the caller must REFUSE".**  This is
        deliberately the OPPOSITE of the dashboard's ``_park_age_seconds``
        (dashboard/src/dashboard/data/scheduler.py), which fails soft to ``0``
        because it is rendering a number for a human and a wrong zero there is
        a harmless display artefact.  Here the age GATES AN ADMISSION (task
        3823 / PRD C7 refuses backfill through a park older than
        ``backfill_max_park_age_secs``), so a soft zero would make an
        unreadable park look brand new and ADMIT a backfiller through it —
        fail-soft in the unsafe direction.  The two contracts must disagree;
        do not "fix" the divergence by consolidating them.

        Reads ``_park_install_at``, which is written once per owner via
        ``setdefault`` — so this is the owner's TOTAL wait, which is what C7's
        ``park_age(p)`` means, not the time since its most recent re-park.

        A stamp in the FUTURE clamps to ``0.0`` rather than going negative:
        a negative age would sail under every ``age > cutoff`` test, so an NTP
        step backwards would silently re-open admission through arbitrarily
        old parks.

        This is a READ.  It never evicts a park — task 1228 removed park
        leases deliberately and nothing here reintroduces wall-clock park
        death.
        """
        raw = self._park_install_at.get(owner)
        if not raw:
            return None
        try:
            installed_at = datetime.fromisoformat(str(raw).replace('Z', '+00:00'))
        except (ValueError, TypeError):
            logger.warning(
                'park_age_secs: unparseable install stamp %r for park owner %s — '
                'returning None (age unknown; callers must refuse)',
                raw, owner,
            )
            return None
        try:
            return max(0.0, (now - installed_at).total_seconds())
        except TypeError:
            # Naive/aware mismatch — same "unknown, refuse" answer as a bad parse.
            logger.warning(
                'park_age_secs: cannot compare install stamp %r for park owner %s '
                'against now=%r — returning None (age unknown)',
                raw, owner, now,
            )
            return None

    def snapshot_parks(self) -> dict[str, dict]:
        """Return a snapshot of current parks: ``{task_id: {modules, installed_at}}``.

        Reports only the ACTIVE-TOP owner of each module stack (INV-7).  Buried
        (shadowed) owners do not appear — the snapshot contract is identical to
        the pre-stack single-owner semantics, keeping the dashboard's
        ``_module_contention_counts`` and fused-memory ``get_scheduler_state``
        working without changes.

        Builds a fresh dict from ``_parked`` and ``_park_install_at`` so callers
        cannot mutate internal state.  Preferred over direct ``_parked`` access in
        :meth:`Scheduler.get_state_snapshot`.
        """
        result: dict[str, dict] = {}
        for module, stack in self._parked.items():
            if not stack:
                continue
            owner, _rank = stack[-1]  # active TOP only (INV-7)
            if owner not in result:
                result[owner] = {
                    'modules': [],
                    'installed_at': self._park_install_at.get(owner, ''),
                }
            result[owner]['modules'].append(module)
        return result

    def snapshot_park_stacks(self) -> dict[str, list[dict]]:
        """Return a snapshot of the full LIFO park stack per module.

        Keys by MODULE (distinct from :meth:`snapshot_parks`, which keys by owner).
        Each value is the stack list in bottom→top order — ``list[-1]`` is the
        active top (the owner that :meth:`snapshot_parks` would report), and
        every preceding element is a shadowed (buried) owner.

        Each entry dict contains:
        - ``owner``: str — task id
        - ``rank``: int — :func:`park_rank` value: a PRIORITY_RANK tier, or a
          negative pin rank (task 6040); lower = higher priority, INV-3
        - ``shadowed``: bool — True for every entry except the active top
        - ``installed_at``: str — ISO8601 timestamp from ``_park_install_at``,
          or ``''`` if the owner has no recorded install timestamp
        - ``source``: ``'pin'`` | ``'fairness'`` — derived from ``rank`` by
          :func:`~orchestrator.pin_reservation.rank_payload`
        - ``pin_order``: int (pin entries only) / ``tier``: str (fairness
          entries only) — the priority behind ``rank``

        INV-3: ranks are non-increasing top-ward (``entry[i].rank >= entry[i+1].rank``),
        and equal ranks are ordered by the id tie-break of :class:`_ParkKey`.

        Builds fresh dicts/lists from ``_parked`` and ``_park_install_at`` so
        callers cannot mutate internal state.
        """
        result: dict[str, list[dict]] = {}
        for module, stack in self._parked.items():
            if not stack:
                continue
            last_index = len(stack) - 1
            entries: list[dict] = []
            for idx, (owner, rank) in enumerate(stack):
                entries.append({
                    'owner': owner,
                    'rank': rank,
                    'shadowed': idx != last_index,
                    'installed_at': self._park_install_at.get(owner, ''),
                    **rank_payload(rank),
                })
            result[module] = entries
        return result

    def snapshot_pin_reservations(self) -> dict[str, dict]:
        """``{owner: {modules, installed_at}}`` for every pin-ranked park entry.

        Unlike :meth:`snapshot_parks`, a buried pin entry is listed too: this
        is "which modules does each pin hold a reservation on" (task 6040),
        not "who tops each stack".  Tier-ranked owners never appear.  Fresh
        dicts, sorted modules.
        """
        result: dict[str, dict] = {}
        for module, stack in self._parked.items():
            for owner, rank in stack:
                if not is_pin_rank(rank):
                    continue
                entry = result.setdefault(owner, {
                    'modules': [],
                    'installed_at': self._park_install_at.get(owner, ''),
                })
                entry['modules'].append(module)
        for entry in result.values():
            entry['modules'].sort()
        return result

    def snapshot_holders(self) -> dict[str, str]:
        """Return a snapshot of current lock holders: ``{module: task_id}``.

        Builds a fresh dict from ``_held`` so callers cannot mutate internal state.
        Preferred over direct ``_held`` access in :meth:`Scheduler.get_state_snapshot`.
        It shows ONE holder per key: a key several tasks hold names only the
        last one iterated, so ask :meth:`held_modules` for what one task holds.
        """
        result: dict[str, str] = {}
        for task_id, modules in self._held.items():
            for m in modules:
                result[m] = task_id
        return result

    # --- Limit lookup ---

    def _limit_for(self, module: str) -> int:
        module = normalize_lock(module, self._config.lock_depth)
        if module not in self._limits:
            self._limits[module] = self._resolve_limit(module)
        return self._limits[module]

    def _resolve_limit(self, key: str) -> int:
        """*key*'s holder limit, the first that applies, in this order (the one statement of it).

        The key's module config's ``module_overrides[key]``; the top-level
        ``module_overrides[key]``; ``UNLIMITED_HOLDERS`` when the top-level
        ``unlimited_holder_locks`` rule takes *key*; the module config's
        ``max_per_module``; the top-level ``max_per_module``.  An exact entry
        beating the rule is how a project keeps one file under a prefix exclusive.
        """
        mc = self._config.for_module(key)
        if mc and mc.module_overrides and key in mc.module_overrides:
            return mc.module_overrides[key]
        if key in self._config.module_overrides:
            return self._config.module_overrides[key]
        if self._config.unlimited_holder_locks.takes_unlimited_holders(key):
            return UNLIMITED_HOLDERS
        if mc and mc.max_per_module is not None:
            return mc.max_per_module
        return self._config.max_per_module

    # --- Public API ---

    def is_held(self, task_id: str) -> bool:
        """Return True if task_id currently holds any module locks."""
        return task_id in self._held

    def try_acquire(
        self,
        task_id: str,
        modules: list[str],
        *,
        admitted_parks: frozenset[str] = frozenset(),
    ) -> bool:
        """Non-blocking attempt to acquire all module locks.

        Uses hierarchical conflict detection: a lock on ``A/B`` conflicts with
        ``A/B/C`` (and vice-versa) but NOT with ``A/D``.  Also refuses if any
        requested module hierarchically conflicts with an active reservation
        owned by a different task (see ``install_parks``) under the capacity
        rule of ``_iter_blocking_park_owners``.

        Returns True if all acquired, False if any unavailable.

        *admitted_parks* (task 3823 / PRD task η, C7) names park owners whose
        reservations this request has been ADMITTED THROUGH by EASY-backfill:
        their parks are ignored by the park gate below, and by nothing else.

        **The held-lock gate is deliberately untouched, and this is the whole
        safety property of C7 (INV-3).**  The limit check below is evaluated on
        LIVE state at call time, inside this method, on every call — admitted
        or not.  Admission never pre-computes a "c's modules are otherwise
        free" set and then acquires on that basis, because a stale scoring-time
        snapshot used as the ACTING basis is precisely the failure class INV-3
        exists to prevent.  So an admitted candidate whose modules turn out to
        be genuinely contended simply fails here and does not dispatch; the
        bypass can only ever waive a PARK, never a real holder.

        Admission also does not disturb the parks it borrows through: nothing
        here removes, shadows or re-ranks them.  Backfill borrows a gap the
        owner was going to wait anyway — PRD §7 rejects preemption of a held
        lock, and this is the mirror-image discipline for parks.
        """
        depth = self._config.lock_depth
        normalized = list({normalize_lock(m, depth) for m in modules})

        # Check every requested module against all other tasks' held locks and
        # active reservations owned by other tasks.
        for module in normalized:
            # INV-3: live held-lock state, never bypassed by admitted_parks.
            if self._count_conflicts(module, exclude_task=task_id) >= self._limit_for(module):
                return False
            if self._is_parked_blocks(module, task_id, ignore_owners=admitted_parks):
                return False

        self._held[task_id] = set(normalized)
        logger.info(f'Task {task_id} acquired locks: {normalized}')
        return True

    def release(self, task_id: str) -> list[str]:
        """Release all module locks held by a task. Returns the normalized
        modules released (empty when the task held none), the same shape as
        ``release_subset``.
        """
        released = sorted(self._held.pop(task_id, set()))
        if released:
            logger.info(f'Task {task_id} released locks: {released}')
        return released

    def release_subset(self, task_id: str, modules: list[str]) -> list[str]:
        """Drop a subset of the task's held modules. Returns the normalized
        modules actually released (may be empty). Removes the task's entry
        entirely when no held modules remain so downstream iteration over
        ``_held`` behaves the same as after a full ``release``.
        """
        held = self._held.get(task_id)
        if not held:
            return []
        depth = self._config.lock_depth
        to_drop = {normalize_lock(m, depth) for m in modules} & held
        if not to_drop:
            return []
        held.difference_update(to_drop)
        if not held:
            del self._held[task_id]
        released = sorted(to_drop)
        logger.info(f'Task {task_id} released subset: {released}')
        return released

    def try_acquire_additional(self, task_id: str, additional: list[str]) -> bool:
        """Non-blocking attempt to expand lock set for a task."""
        depth = self._config.lock_depth
        current = self._held.get(task_id, set())
        new_modules = [
            normalize_lock(m, depth)
            for m in additional
            if normalize_lock(m, depth) not in current
        ]
        if not new_modules:
            return True

        for module in new_modules:
            if self._count_conflicts(module, exclude_task=task_id) >= self._limit_for(module):
                return False
            if self._is_parked_blocks(module, task_id):
                return False

        self._held.setdefault(task_id, set()).update(new_modules)
        logger.info(f'Task {task_id} expanded locks: {new_modules}')
        return True

    def held_modules(self, task_id: str) -> frozenset[str]:
        """The modules *task_id* holds now, as an immutable snapshot. A READ."""
        return frozenset(self._held.get(task_id, ()))

    def acquire_uncontested(
        self, task_id: str, modules: Iterable[str]
    ) -> UncontestedAcquisition:
        """Take every free key of *modules*, name the rest, release nothing.

        Keys the task already holds are skipped. The contested keys are
        exactly :meth:`blockers`, which names what the acquire gate honours
        (INV-5), so the free remainder must then be acquirable; a refusal is
        an invariant breach and raises rather than returning a silent partial.
        Never releases a lock and never touches task status.
        """
        depth = self._config.lock_depth
        held = self.held_modules(task_id)
        requested = [
            key
            for key in dict.fromkeys(normalize_lock(m, depth) for m in modules)
            if key and key not in held
        ]
        contested = tuple(self.blockers(task_id, requested))
        contested_keys = {blocker.module for blocker in contested}
        free = [key for key in requested if key not in contested_keys]
        if free and not self.try_acquire_additional(task_id, free):
            raise RuntimeError(
                f'Task {task_id}: try_acquire_additional refused {free}, which '
                'blockers() reported free; INV-5 requires the two to agree'
            )
        return UncontestedAcquisition(acquired=tuple(free), contested=contested)
