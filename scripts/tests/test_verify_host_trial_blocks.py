"""verify_host_trial_blocks.derive_blocks: the on/off timeline of one verify host."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from verify_host_trial_blocks import (
    BLOCK_RULE,
    Block,
    BlockOrigin,
    HostState,
    LifecycleOrigin,
    Transition,
    derive_blocks,
)
from verify_host_trial_stores import EventType

SINCE = datetime(2026, 11, 11, tzinfo=UTC)
UNTIL = datetime(2026, 11, 25, tzinfo=UTC)
ON, OFF = HostState.ON, HostState.OFF


def day(n: float) -> datetime:
    return SINCE + timedelta(days=n)


def retired(at: datetime) -> Transition:
    return Transition(at, OFF, LifecycleOrigin.RETIRED)


def admitted(at: datetime) -> Transition:
    return Transition(at, ON, LifecycleOrigin.ADMITTED)


def shape(blocks: tuple[Block, ...]) -> list[tuple[HostState, datetime, datetime, BlockOrigin, datetime]]:
    return [(b.state, b.start, b.end, b.opened_by, b.opened_at) for b in blocks]


def assert_tiles(blocks: tuple[Block, ...], since: datetime = SINCE, until: datetime = UNTIL) -> None:
    assert blocks[0].start == since
    assert blocks[-1].end == until
    for left, right in zip(blocks, blocks[1:], strict=False):
        assert left.end == right.start
    assert all(b.start < b.end for b in blocks)


class TestBeforeAnyLifecycleRow:
    def test_a_host_verifying_before_the_window_is_on_throughout(self):
        """The live laptop shape: no lifecycle row, first merge_verify long before --since."""
        first_verify = datetime(2026, 10, 1, 9, tzinfo=UTC)

        blocks = derive_blocks((), first_verify, SINCE, UNTIL)

        assert shape(blocks) == [(ON, SINCE, UNTIL, BlockOrigin.FIRST_VERIFY, first_verify)]

    def test_a_first_verify_inside_the_window_splits_it(self):
        blocks = derive_blocks((), day(4), SINCE, UNTIL)

        assert shape(blocks) == [
            (OFF, SINCE, day(4), BlockOrigin.WINDOW_START, SINCE),
            (ON, day(4), UNTIL, BlockOrigin.FIRST_VERIFY, day(4)),
        ]

    def test_a_host_with_no_row_at_all_is_off_throughout(self):
        blocks = derive_blocks((), None, SINCE, UNTIL)

        assert shape(blocks) == [(OFF, SINCE, UNTIL, BlockOrigin.WINDOW_START, SINCE)]


class TestLifecycleRows:
    def test_toggles_inside_the_window_alternate_blocks_each_opened_by_its_row(self):
        first_verify = day(-30)
        rows = (retired(day(3)), admitted(day(6)), retired(day(9)))

        blocks = derive_blocks(rows, first_verify, SINCE, UNTIL)

        assert shape(blocks) == [
            (ON, SINCE, day(3), BlockOrigin.FIRST_VERIFY, first_verify),
            (OFF, day(3), day(6), BlockOrigin.RETIRED, day(3)),
            (ON, day(6), day(9), BlockOrigin.ADMITTED, day(6)),
            (OFF, day(9), UNTIL, BlockOrigin.RETIRED, day(9)),
        ]

    def test_one_toggle_reloaded_in_several_stores_is_one_boundary_at_its_earliest_row(self):
        toggle_off = day(3)
        toggle_on = day(6)
        rows = (
            admitted(toggle_on + timedelta(seconds=40)),
            retired(toggle_off + timedelta(seconds=90)),
            retired(toggle_off),
            admitted(toggle_on),
            retired(toggle_off + timedelta(seconds=5)),
        )

        blocks = derive_blocks(rows, day(-30), SINCE, UNTIL)

        assert shape(blocks) == [
            (ON, SINCE, toggle_off, BlockOrigin.FIRST_VERIFY, day(-30)),
            (OFF, toggle_off, toggle_on, BlockOrigin.RETIRED, toggle_off),
            (ON, toggle_on, UNTIL, BlockOrigin.ADMITTED, toggle_on),
        ]

    def test_a_row_before_the_window_sets_the_state_at_since(self):
        blocks = derive_blocks((retired(day(-2)),), None, SINCE, UNTIL)

        assert shape(blocks) == [(OFF, SINCE, UNTIL, BlockOrigin.RETIRED, day(-2))]

    def test_rows_before_the_window_after_a_first_verify_keep_the_latest_state(self):
        rows = (retired(day(-5)), admitted(day(-2)))

        blocks = derive_blocks(rows, day(-30), SINCE, UNTIL)

        assert shape(blocks) == [(ON, SINCE, UNTIL, BlockOrigin.ADMITTED, day(-2))]

    def test_a_retired_row_with_no_verify_before_it_leaves_the_host_off_throughout(self):
        """The host never verified, so the retirement changes nothing."""
        blocks = derive_blocks((retired(day(5)),), None, SINCE, UNTIL)

        assert shape(blocks) == [(OFF, SINCE, UNTIL, BlockOrigin.WINDOW_START, SINCE)]


class TestTiling:
    @pytest.mark.parametrize(
        ('rows', 'first_verify'),
        [
            ((retired(SINCE),), day(-30)),
            ((retired(UNTIL),), day(-30)),
            ((), SINCE),
            ((), UNTIL),
            ((retired(SINCE), admitted(UNTIL)), day(-30)),
        ],
        ids=['retired-at-since', 'retired-at-until', 'verify-at-since', 'verify-at-until', 'both-edges'],
    )
    def test_a_boundary_at_either_edge_makes_no_zero_length_block(self, rows, first_verify):
        blocks = derive_blocks(rows, first_verify, SINCE, UNTIL)

        assert len(blocks) == 1
        assert_tiles(blocks)

    def test_a_boundary_at_since_opens_the_first_block(self):
        blocks = derive_blocks((retired(SINCE),), day(-30), SINCE, UNTIL)

        assert shape(blocks) == [(OFF, SINCE, UNTIL, BlockOrigin.RETIRED, SINCE)]

    def test_blocks_tile_the_window_exactly(self):
        rows = (retired(day(1)), admitted(day(2.5)), retired(day(7)), admitted(day(13.75)))

        blocks = derive_blocks(rows, day(0.5), SINCE, UNTIL)

        assert_tiles(blocks)
        assert sum(b.days for b in blocks) == pytest.approx(14.0)

    @pytest.mark.parametrize('until', [SINCE, SINCE - timedelta(days=1)], ids=['equal', 'before'])
    def test_an_empty_or_reversed_window_is_refused_naming_both_bounds(self, until):
        with pytest.raises(ValueError) as refusal:
            derive_blocks((), None, SINCE, until)

        assert SINCE.isoformat() in str(refusal.value)
        assert until.isoformat() in str(refusal.value)


class TestBlock:
    def test_days_and_half_open_containment(self):
        block = Block(ON, day(1), day(2.5), BlockOrigin.ADMITTED, day(1))

        assert block.days == 1.5
        assert block.contains(day(1))
        assert block.contains(day(2.5) - timedelta(microseconds=1))
        assert not block.contains(day(2.5))
        assert not block.contains(day(1) - timedelta(microseconds=1))


def test_the_block_rule_names_the_event_types_the_blocks_are_read_from():
    for event_type in (EventType.VERIFY_HOST_ADMITTED, EventType.VERIFY_HOST_RETIRED, EventType.MERGE_VERIFY):
        assert event_type.value in BLOCK_RULE, event_type
