"""The unfiled check: reports read on the inbox wait until a carrier exists or a ruling rules them, and the heartbeat names `filer` once one has waited past the grace (PRD §14.2 decisions 8 and 14)."""

from collections.abc import Set as AbstractSet
from datetime import UTC, datetime, timedelta

import pytest
from _defects import reported, ruled

from client_feed.contract import Degradation, InboxLine, RevealRequested, Source
from client_feed.defect_ref import portal_ref
from client_feed.unfiled import (
    deferred_gate_degradation,
    still_behind_deferred_gates,
    still_unfiled,
    unfiled_degradation,
)

PROJECT = 'acme'
EPOCH = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
GRACE = 600


def at(seconds: float) -> datetime:
    return EPOCH + timedelta(seconds=seconds)


def requested(path: str = 'src/a.py', *, seq: int = 3) -> RevealRequested:
    return RevealRequested(seq=seq, at=at(0), path=path)


def unfiled(
    held: dict[str, datetime],
    *lines: InboxLine,
    filed: AbstractSet[str] | None = frozenset(),
) -> dict[str, datetime]:
    return still_unfiled(held, lines, project_id=PROJECT, filed=filed)


REF = portal_ref(PROJECT, 2, 1)
OTHER_REF = portal_ref(PROJECT, 3, 4)


class TestStillUnfiled:
    def test_a_report_with_no_carrier_waits_from_when_it_was_reported(self) -> None:
        line = reported(2, 1, at=at(30))

        assert unfiled({}, line) == {REF: line.at}

    def test_a_held_report_keeps_its_first_time_when_it_is_read_again(self) -> None:
        assert unfiled({REF: at(0)}, reported(2, 1, at=at(90))) == {REF: at(0)}

    def test_a_filed_ref_is_dropped_whether_held_or_newly_read(self) -> None:
        held = {REF: at(0)}

        assert unfiled(held, reported(3, 4, at=at(5)), filed={REF, OTHER_REF}) == {}

    def test_unreadable_carriers_drop_nothing(self) -> None:
        held = {REF: at(0)}

        assert unfiled(held, reported(3, 4, at=at(5)), filed=None) == {
            REF: at(0),
            OTHER_REF: at(5),
        }

    def test_a_ruling_drops_a_held_report(self) -> None:
        assert unfiled({REF: at(0), OTHER_REF: at(1)}, ruled(2, 1, at=at(9))) == {
            OTHER_REF: at(1)
        }

    def test_a_ruling_drops_a_report_read_earlier_in_the_same_lines(self) -> None:
        lines = (reported(2, 1, at=at(3), seq=1), ruled(2, 1, at=at(4), seq=2))

        assert unfiled({}, *lines, filed=None) == {}

    def test_reveal_requests_change_nothing(self) -> None:
        held = {REF: at(0)}

        assert unfiled(held, requested(), requested('src/b.py', seq=4)) == held

    def test_no_lines_and_nothing_filed_hand_back_the_held_reports(self) -> None:
        held = {REF: at(0), OTHER_REF: at(7)}

        assert unfiled(held) == held

    def test_the_held_map_is_never_mutated(self) -> None:
        held = {REF: at(0)}

        unfiled(held, reported(3, 4, at=at(0)), ruled(2, 1, at=at(0)))

        assert held == {REF: at(0)}


class TestUnfiledDegradation:
    def test_nothing_pending_names_nothing(self) -> None:
        assert unfiled_degradation({}, now=at(10_000), grace_seconds=GRACE) is None

    def test_a_report_exactly_the_grace_old_is_not_yet_overdue(self) -> None:
        assert unfiled_degradation({REF: at(0)}, now=at(GRACE), grace_seconds=GRACE) is None

    def test_a_report_older_than_the_grace_names_the_filer(self) -> None:
        degradation = unfiled_degradation({REF: at(0)}, now=at(GRACE + 1), grace_seconds=GRACE)

        assert degradation == Degradation(
            source=Source.FILER, reason='1 defect report(s) unfiled past 600s: portal:acme:2:1'
        )

    def test_overdue_refs_are_named_oldest_first_and_a_young_one_is_not(self) -> None:
        pending = {
            'portal:acme:2:1': at(200),
            'portal:acme:9:9': at(100),
            'portal:acme:5:5': at(2_000),
        }

        degradation = unfiled_degradation(pending, now=at(2_100), grace_seconds=GRACE)

        assert degradation == Degradation(
            source=Source.FILER,
            reason='2 defect report(s) unfiled past 600s: portal:acme:9:9, portal:acme:2:1',
        )

    def test_overdue_refs_reported_at_one_time_are_named_in_ref_order(self) -> None:
        pending = {'portal:acme:3:1': at(0), 'portal:acme:2:1': at(0)}

        degradation = unfiled_degradation(pending, now=at(GRACE + 1), grace_seconds=GRACE)

        assert degradation is not None
        assert degradation.reason.endswith(': portal:acme:2:1, portal:acme:3:1')

    @pytest.mark.parametrize('grace_seconds', [0, 60, 3_600])
    def test_the_reason_names_the_grace_it_was_judged_against(self, grace_seconds: int) -> None:
        degradation = unfiled_degradation(
            {REF: at(0)}, now=at(grace_seconds + 1), grace_seconds=grace_seconds
        )

        assert degradation is not None
        assert f' past {grace_seconds}s: ' in degradation.reason


def behind(
    held: dict[str, datetime], waiting: AbstractSet[str] | None, *, now: datetime = at(500)
) -> dict[str, datetime]:
    return still_behind_deferred_gates(held, waiting, now=now)


class TestStillBehindDeferredGates:
    def test_a_ref_newly_waiting_is_held_since_now(self) -> None:
        assert behind({}, {REF}, now=at(500)) == {REF: at(500)}

    def test_a_held_ref_still_waiting_keeps_its_first_time(self) -> None:
        assert behind({REF: at(10)}, {REF}, now=at(500)) == {REF: at(10)}

    def test_a_held_ref_no_longer_waiting_is_dropped(self) -> None:
        assert behind({REF: at(10), OTHER_REF: at(20)}, {OTHER_REF}) == {OTHER_REF: at(20)}

    def test_unreadable_carriers_hand_back_the_held_refs_and_add_nothing(self) -> None:
        assert behind({REF: at(10)}, None, now=at(500)) == {REF: at(10)}

    def test_nothing_waiting_empties_the_map(self) -> None:
        assert behind({REF: at(10), OTHER_REF: at(20)}, frozenset()) == {}

    def test_the_held_map_is_never_mutated(self) -> None:
        held = {REF: at(10)}

        behind(held, {OTHER_REF})
        behind(held, None)

        assert held == {REF: at(10)}


class TestDeferredGateDegradation:
    def test_nothing_held_names_nothing(self) -> None:
        assert deferred_gate_degradation({}, now=at(10_000), grace_seconds=GRACE) is None

    def test_a_ref_held_exactly_the_grace_is_not_yet_overdue(self) -> None:
        assert deferred_gate_degradation({REF: at(0)}, now=at(GRACE), grace_seconds=GRACE) is None

    def test_a_ref_held_longer_than_the_grace_names_the_filer(self) -> None:
        degradation = deferred_gate_degradation(
            {REF: at(0)}, now=at(GRACE + 1), grace_seconds=GRACE
        )

        assert degradation == Degradation(
            source=Source.FILER,
            reason=f'1 defect report(s) waiting on a deferred gate past 600s: {REF}',
        )

    def test_overdue_refs_are_named_oldest_first_ties_in_ref_order_and_a_young_one_is_not(
        self,
    ) -> None:
        held = {
            'portal:acme:2:1': at(200),
            'portal:acme:9:9': at(100),
            'portal:acme:3:3': at(100),
            'portal:acme:5:5': at(2_000),
        }

        degradation = deferred_gate_degradation(held, now=at(2_100), grace_seconds=GRACE)

        assert degradation == Degradation(
            source=Source.FILER,
            reason=(
                '3 defect report(s) waiting on a deferred gate past 600s:'
                ' portal:acme:3:3, portal:acme:9:9, portal:acme:2:1'
            ),
        )

    @pytest.mark.parametrize('grace_seconds', [0, 60, 3_600])
    def test_the_reason_names_the_grace_it_was_judged_against(self, grace_seconds: int) -> None:
        degradation = deferred_gate_degradation(
            {REF: at(0)}, now=at(grace_seconds + 1), grace_seconds=grace_seconds
        )

        assert degradation is not None
        assert f' past {grace_seconds}s: ' in degradation.reason
