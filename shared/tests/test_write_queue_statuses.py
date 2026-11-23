"""The durable write queue's row-status vocabulary and its two partitions."""

from __future__ import annotations

import enum

from shared.write_queue_statuses import AWAITING_OPERATOR, CLAIMABLE, WriteQueueStatus


def test_the_vocabulary_is_every_status_a_write_queue_row_takes():
    assert issubclass(WriteQueueStatus, enum.StrEnum)
    assert {s.value for s in WriteQueueStatus} == {
        'pending', 'retry', 'parked', 'in_flight', 'completed', 'dead',
    }


def test_claimable_is_ordered_because_sql_is_built_from_it():
    assert CLAIMABLE == (
        WriteQueueStatus.PENDING, WriteQueueStatus.RETRY, WriteQueueStatus.PARKED,
    )


def test_only_an_operator_moves_a_parked_or_dead_row_on():
    assert frozenset({WriteQueueStatus.PARKED, WriteQueueStatus.DEAD}) == AWAITING_OPERATOR


def test_a_member_looks_up_the_plain_str_counts_get_stats_emits():
    assert {'parked': 3}.get(WriteQueueStatus.PARKED) == 3
