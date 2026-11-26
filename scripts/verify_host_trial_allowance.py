"""Estimate the allowance each account let expire unused, cycle by cycle, from cap hits and recorded spend.

The stores record neither an account's remaining allowance nor its reset
before it caps. The only grounded figure is spend at the weekly cap, so the
allowance is estimated from weekly-capped cycles, and the estimate is
unknown, never 0, when there is none. Session, per-model and unrecognised
caps place no cycle; they are counted by kind. Cap hits and spend are pooled
over every store read.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import MappingProxyType

from verify_host_trial_blocks import Block
from verify_host_trial_stores import CapHit, CapKind, Invocation

__all__ = ['ALLOWANCE_RULE', 'CYCLE', 'AccountCycle', 'AllowanceReading', 'account_cycles', 'expired_in_block']

CYCLE = timedelta(days=7)
_SAME_CYCLE = timedelta(hours=12)
"""A capped reset this close to a lattice point is that cycle (resets move with clock changes)."""

ALLOWANCE_RULE = (
    'allowance expired unused, an estimate in list USD with every model counted once: an account cycle runs '
    '[resets_at - 7 d, resets_at); only a weekly cap (a cap_hit whose reason names the weekly limit, with no '
    'model scope) marks its cycle capped (one per account and resets_at across stores), expiring nothing, '
    'anchors the lattice and enters the estimate; session or N-hour, per-model and unrecognised caps place '
    'no cycle and are counted by kind (one per account, kind and resets_at) within one cycle of the window; '
    "uncapped cycles sit on a weekly lattice from the account's latest recorded weekly resets_at (a capped "
    'reset within 12 h of a lattice point replaces it), for accounts with spend or a cap within one cycle of '
    'the window, the others listed as dormant; the allowance estimate is the median recorded spend, over '
    'every store read, of the capped cycles resetting in the window; an uncapped cycle expired max(0, '
    'estimate - its spend), counted in the block holding its reset; with no capped cycle the estimate and '
    'every expiry are unknown'
)


@dataclass(frozen=True)
class AccountCycle:
    account: str
    reset: datetime
    capped: bool
    capped_at: datetime | None
    spend_usd: float
    expired_usd: float | None
    """0 for a capped cycle; None when the allowance estimate is unknown."""

    @property
    def start(self) -> datetime:
        return self.reset - CYCLE

    @property
    def hours_capped(self) -> float | None:
        return None if self.capped_at is None else (self.reset - self.capped_at).total_seconds() / 3600


@dataclass(frozen=True)
class AllowanceReading:
    cycles: tuple[AccountCycle, ...]
    """Every placed cycle resetting in the window, in reset order."""
    estimate_usd: float | None
    unplaced_accounts: tuple[str, ...]
    """Accounts with spend but no weekly reset ever recorded."""
    dormant_accounts: tuple[str, ...]
    """Accounts with a recorded weekly reset but no spend or cap within one cycle of the window."""
    unplaceable_cap_hits: int
    """Weekly cap hits within one cycle of the window recorded without a reset."""
    other_caps: Mapping[CapKind, int]
    """Distinct non-weekly caps within one cycle of the window, by kind; only kinds seen are present."""


def _near(at: datetime, since: datetime, until: datetime) -> bool:
    """Whether *at* lies within one cycle of the window [since, until)."""
    return since - CYCLE <= at < until


def _first_caps(cap_hits: Iterable[CapHit]) -> dict[tuple[str, datetime], datetime]:
    """Each weekly-capped (account, reset), with its earliest weekly cap across stores."""
    caps: dict[tuple[str, datetime], datetime] = {}
    for hit in cap_hits:
        if hit.kind is CapKind.WEEKLY and hit.resets_at is not None:
            key = (hit.account, hit.resets_at)
            caps[key] = min(hit.at, caps.get(key, hit.at))
    return caps


def _lattice(latest: datetime, since: datetime, until: datetime) -> list[datetime]:
    """The resets in [since, until) a weekly step away from *latest*."""
    point = latest + math.ceil((since - latest) / CYCLE) * CYCLE
    while point < since:
        point += CYCLE
    points = []
    while point < until:
        points.append(point)
        point += CYCLE
    return points


def _placed_resets(
    caps: Mapping[tuple[str, datetime], datetime], account: str, latest: datetime, since: datetime, until: datetime
) -> list[tuple[datetime, datetime | None]]:
    """*account*'s resets in the window, each with its cap time or None for an uncapped one."""
    capped = {reset: at for (acct, reset), at in caps.items() if acct == account and since <= reset < until}
    uncapped = [
        point for point in _lattice(latest, since, until)
        if all(abs(point - reset) > _SAME_CYCLE for reset in capped)
    ]
    return [*capped.items(), *((point, None) for point in uncapped)]


def _other_caps(cap_hits: Iterable[CapHit], since: datetime, until: datetime) -> Mapping[CapKind, int]:
    """Distinct non-weekly caps near the window by kind; a cap without a reset is keyed on its own time."""
    distinct = {
        (hit.account, hit.kind, hit.resets_at, hit.at if hit.resets_at is None else None)
        for hit in cap_hits
        if hit.kind is not CapKind.WEEKLY and _near(hit.at, since, until)
    }
    counts = Counter(kind for _account, kind, _reset, _at in distinct)
    return MappingProxyType({kind: counts[kind] for kind in CapKind if counts[kind]})


def _cycle_spend(invocations: Sequence[Invocation], reset: datetime) -> float:
    return sum(i.cost_usd for i in invocations if reset - CYCLE <= i.completed_at < reset)


def _expiry(capped: bool, spent: float, estimate: float | None) -> float | None:
    if capped:
        return 0.0
    return None if estimate is None else max(0.0, estimate - spent)


def account_cycles(
    cap_hits: Iterable[CapHit], spend: Iterable[Invocation], since: datetime, until: datetime
) -> AllowanceReading:
    """Every account cycle resetting in [since, until), with its spend and its estimated expiry."""
    cap_hits, spend = tuple(cap_hits), tuple(spend)
    caps = _first_caps(cap_hits)
    latest: dict[str, datetime] = {}
    for account, reset in caps:
        latest[account] = max(reset, latest.get(account, reset))
    by_account: dict[str, list[Invocation]] = defaultdict(list)
    for invocation in spend:
        by_account[invocation.account].append(invocation)
    active = {i.account for i in spend if _near(i.completed_at, since, until)}
    active |= {hit.account for hit in cap_hits if _near(hit.at, since, until)}
    placed = [
        (account, reset, capped_at, _cycle_spend(by_account[account], reset))
        for account in sorted(latest.keys() & active)
        for reset, capped_at in _placed_resets(caps, account, latest[account], since, until)
    ]
    capped_spends = [spent for _account, _reset, capped_at, spent in placed if capped_at is not None]
    estimate = statistics.median(capped_spends) if capped_spends else None
    cycles = tuple(
        AccountCycle(account, reset, capped_at is not None, capped_at, spent, _expiry(capped_at is not None, spent, estimate))
        for account, reset, capped_at, spent in sorted(placed, key=lambda row: (row[1], row[0]))
    )
    return AllowanceReading(
        cycles=cycles,
        estimate_usd=estimate,
        unplaced_accounts=tuple(sorted(by_account.keys() - latest.keys())),
        dormant_accounts=tuple(sorted(latest.keys() - active)),
        unplaceable_cap_hits=sum(
            1 for hit in cap_hits if hit.kind is CapKind.WEEKLY and hit.resets_at is None and _near(hit.at, since, until)
        ),
        other_caps=_other_caps(cap_hits, since, until),
    )


def expired_in_block(reading: AllowanceReading, block: Block) -> float | None:
    """The expiry of the cycles resetting in *block*; None if any of them is unknown."""
    total = 0.0
    for cycle in reading.cycles:
        if block.contains(cycle.reset):
            if cycle.expired_usd is None:
                return None
            total += cycle.expired_usd
    return total
