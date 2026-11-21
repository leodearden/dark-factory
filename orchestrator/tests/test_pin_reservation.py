"""The rank encoding in orchestrator.pin_reservation and its inverse.

``park_rank`` turns a park priority into a stack rank; ``rank_payload`` and
``PinOrder.from_rank`` name the priority behind a stored rank.  The inverse
must round-trip every rank ``park_rank`` can produce and refuse every rank it
cannot, so a breach of the band invariant is reported instead of relabelled.
"""

from __future__ import annotations

import pytest

from orchestrator.config import PRIORITY_TIERS
from orchestrator.pin_reservation import PinOrder, park_rank, rank_payload


@pytest.mark.parametrize('pin_order', [0, 1, 42, 999_998])
def test_from_rank_inverts_pin_order_rank(pin_order):
    assert PinOrder.from_rank(PinOrder(pin_order).rank) == PinOrder(pin_order)


@pytest.mark.parametrize('rank', [PinOrder(0).rank - 1, park_rank(PRIORITY_TIERS[0])])
def test_from_rank_refuses_rank_outside_pin_band(rank):
    with pytest.raises(ValueError, match=f'rank {rank} '):
        PinOrder.from_rank(rank)


@pytest.mark.parametrize('tier', PRIORITY_TIERS)
def test_rank_payload_names_every_tier(tier):
    assert rank_payload(park_rank(tier)) == {'source': 'fairness', 'tier': tier}


def test_rank_payload_names_pin_order():
    assert rank_payload(park_rank(PinOrder(3))) == {'source': 'pin', 'pin_order': 3}


def test_rank_payload_refuses_rank_past_last_tier():
    rank = park_rank(PRIORITY_TIERS[-1]) + 1
    with pytest.raises(ValueError, match=f'rank {rank} names no priority tier'):
        rank_payload(rank)
