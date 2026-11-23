"""The ``park_refused`` class predicate and its per-task episode cadence.

``is_park_refusal`` decides whether a refused candidate is waiting on parks
alone, none of which outranks it; ``ParkRefusalEpisodes`` decides when such a
wait is old enough, and again old enough, to be reported.
"""

from __future__ import annotations

import pytest

from orchestrator.park_refusal import ParkRefusalEpisodes, is_park_refusal
from orchestrator.pin_reservation import Blocker, BlockerKind, PinOrder, park_rank


def _parked(module: str, owner: str) -> Blocker:
    return Blocker(module, owner, BlockerKind.PARKED)


def _held(module: str, owner: str) -> Blocker:
    return Blocker(module, owner, BlockerKind.HELD)


class TestIsParkRefusal:
    def test_no_blockers_is_not_a_park_refusal(self):
        assert not is_park_refusal('critical', [], {}.get)

    def test_a_held_blocker_beside_parked_ones_is_not_a_park_refusal(self):
        ranks = {'A': park_rank('critical')}
        blockers = [_parked('x.py', 'A'), _held('y.py', 'H')]
        assert not is_park_refusal('critical', blockers, ranks.get)

    def test_a_same_tier_park_is_a_park_refusal(self):
        ranks = {'A': park_rank('critical')}
        assert is_park_refusal('critical', [_parked('x.py', 'A')], ranks.get)

    def test_a_weaker_tier_park_is_a_park_refusal(self):
        ranks = {'L': park_rank('medium')}
        assert is_park_refusal('critical', [_parked('x.py', 'L')], ranks.get)

    def test_a_stronger_tier_park_is_not_a_park_refusal(self):
        ranks = {'A': park_rank('critical')}
        assert not is_park_refusal('high', [_parked('x.py', 'A')], ranks.get)

    @pytest.mark.parametrize('tier', ['critical', 'high', 'medium', 'low', 'polish'])
    def test_a_pin_rank_owner_is_never_a_park_refusal(self, tier):
        ranks = {'P': PinOrder(1).rank}
        assert not is_park_refusal(tier, [_parked('x.py', 'P')], ranks.get)

    def test_an_owner_topping_no_stack_is_not_a_park_refusal(self):
        assert not is_park_refusal('critical', [_parked('x.py', 'gone')], {}.get)

    def test_one_outranking_owner_among_two_is_not_a_park_refusal(self):
        ranks = {'L': park_rank('low'), 'A': park_rank('critical')}
        blockers = [_parked('x.py', 'L'), _parked('y.py', 'A')]
        assert not is_park_refusal('high', blockers, ranks.get)


class TestParkRefusalEpisodes:
    INTERVAL = 100.0

    def test_first_observation_starts_the_episode_without_emitting(self):
        episodes = ParkRefusalEpisodes()
        assert episodes.observe('B', now=10.0, interval=self.INTERVAL) is None

    def test_age_is_measured_from_start_and_cadence_from_last_emission(self):
        episodes = ParkRefusalEpisodes()
        t0, i = 10.0, self.INTERVAL
        assert episodes.observe('B', now=t0, interval=i) is None
        assert episodes.observe('B', now=t0 + i - 1, interval=i) is None
        assert episodes.observe('B', now=t0 + i, interval=i) == i
        assert episodes.observe('B', now=t0 + 2 * i - 1, interval=i) is None
        assert episodes.observe('B', now=t0 + 2 * i, interval=i) == 2 * i

    def test_end_then_observe_starts_a_new_episode(self):
        episodes = ParkRefusalEpisodes()
        t0, i = 10.0, self.INTERVAL
        episodes.observe('B', now=t0, interval=i)
        assert episodes.observe('B', now=t0 + i, interval=i) == i
        episodes.end('B')
        t1 = t0 + i + 5
        assert episodes.observe('B', now=t1, interval=i) is None
        assert episodes.observe('B', now=t1 + i - 1, interval=i) is None
        assert episodes.observe('B', now=t1 + i, interval=i) == i

    def test_retain_without_the_task_ends_its_episode(self):
        episodes = ParkRefusalEpisodes()
        t0, i = 10.0, self.INTERVAL
        episodes.observe('B', now=t0, interval=i)
        episodes.retain(['other'])
        assert episodes.observe('B', now=t0 + i, interval=i) is None
        assert episodes.observe('B', now=t0 + 2 * i, interval=i) == i

    def test_retain_with_the_task_keeps_its_episode(self):
        episodes = ParkRefusalEpisodes()
        t0, i = 10.0, self.INTERVAL
        episodes.observe('B', now=t0, interval=i)
        episodes.retain(['B', 'other'])
        assert episodes.observe('B', now=t0 + i, interval=i) == i

    def test_two_tasks_keep_independent_episodes(self):
        episodes = ParkRefusalEpisodes()
        i = self.INTERVAL
        episodes.observe('B', now=0.0, interval=i)
        episodes.observe('C', now=50.0, interval=i)
        assert episodes.observe('B', now=i, interval=i) == i
        assert episodes.observe('C', now=i, interval=i) is None
        assert episodes.observe('C', now=50.0 + i, interval=i) == i
        episodes.end('B')
        assert episodes.observe('C', now=50.0 + 2 * i, interval=i) == 2 * i

    def test_a_smaller_interval_on_a_later_call_applies_at_once(self):
        episodes = ParkRefusalEpisodes()
        episodes.observe('B', now=0.0, interval=1000.0)
        assert episodes.observe('B', now=60.0, interval=1000.0) is None
        assert episodes.observe('B', now=60.0, interval=60.0) == 60.0
