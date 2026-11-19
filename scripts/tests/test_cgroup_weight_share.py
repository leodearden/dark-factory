"""Tests for scripts/cgroup_weight_share.py: the weight-fair share a cgroup is entitled to."""
from __future__ import annotations

import dataclasses

import pytest
from cgroup_weight_share import Claim, fair_shares


def test_every_claimant_oversubscribed_splits_by_weight():
    claims = {'a': Claim(100, 20.0), 'b': Claim(100, 20.0), 'c': Claim(100, 20.0)}
    assert fair_shares(12.0, claims) == pytest.approx({'a': 4.0, 'b': 4.0, 'c': 4.0})


def test_merge_and_task_scopes_split_100_to_33():
    claims = {'merge': Claim(100, 20.0), 'task': Claim(33, 20.0)}
    assert fair_shares(13.3, claims) == pytest.approx({'merge': 10.0, 'task': 3.3})


def test_satisfied_claimant_gets_its_demand_and_rest_is_resplit_by_weight():
    claims = {'a': Claim(100, 1.0), 'b': Claim(100, 20.0), 'c': Claim(100, 20.0)}
    assert fair_shares(12.0, claims) == pytest.approx({'a': 1.0, 'b': 5.5, 'c': 5.5})


@pytest.mark.parametrize('capacity', [10.0, 32.0])
def test_capacity_covering_total_demand_gives_everyone_their_demand(capacity):
    claims = {'a': Claim(100, 2.0), 'b': Claim(33, 3.0), 'c': Claim(10, 5.0)}
    assert fair_shares(capacity, claims) == pytest.approx({'a': 2.0, 'b': 3.0, 'c': 5.0})


def test_idle_claimant_gets_nothing_and_does_not_dilute_the_rest():
    claims = {'idle': Claim(100, 0.0), 'merge': Claim(100, 20.0), 'task': Claim(33, 20.0)}
    assert fair_shares(13.3, claims) == pytest.approx({'idle': 0.0, 'merge': 10.0, 'task': 3.3})


def test_any_unweighted_claimant_means_no_entitlement_exists():
    claims = {'a': Claim(100, 20.0), 'b': Claim(None, 20.0)}
    assert fair_shares(12.0, claims) is None


def test_no_claimants_yields_empty_shares():
    assert fair_shares(12.0, {}) == {}


def test_claim_is_frozen():
    claim = Claim(100, 1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        claim.weight = 33  # type: ignore[misc]
