"""What Scheduler.task_priority answers from between ticks (task 9933)."""
from __future__ import annotations

from pathlib import Path

from orchestrator.overrides import OverrideRow, OverrideSource, OverrideStore
from orchestrator.scheduler_priority_lookup import LookupPins, pin_orders
from orchestrator.task_priority import PINNED, TaskPriority


def _row(*, pinned: bool, pin_order: int | None = None) -> OverrideRow:
    return OverrideRow(
        boost_tier=None, pinned=pinned, pin_order=pin_order, reserve_now=False,
        ttl_until=None, source=OverrideSource.OPERATOR,
    )


class TestPinOrders:

    def test_every_pinned_row_maps_to_its_order(self) -> None:
        rows = {
            'P': _row(pinned=True, pin_order=3),
            'Q': _row(pinned=True, pin_order=None),
            'U': _row(pinned=False, pin_order=None),
        }

        assert pin_orders(rows) == {'P': 3, 'Q': 0}


class TestLookupPins:

    def test_without_a_store_the_pin_is_unknown(self) -> None:
        pins = LookupPins(None, '/project', clock=lambda: 0.0)

        assert pins.priority('X', 'low') == TaskPriority(tier='low', pinned=None, base_tier='low')

    def test_without_a_store_adopting_rows_changes_nothing(self) -> None:
        pins = LookupPins(None, '/project', clock=lambda: 0.0)

        pins.adopt({'X': _row(pinned=True, pin_order=1)})

        assert pins.priority('X', 'low').pinned is None

    def test_an_adopted_pin_answers_pinned_with_its_order(self, tmp_path: Path) -> None:
        pins = LookupPins(OverrideStore(tmp_path / 'overrides.db'), str(tmp_path), clock=lambda: 0.0)

        pins.adopt({'P': _row(pinned=True, pin_order=2), 'U': _row(pinned=False)})

        assert pins.priority('P', 'low') == TaskPriority(
            tier=PINNED, pinned=True, base_tier='low', pin_order=2,
        )
        assert pins.priority('U', 'high') == TaskPriority(
            tier='high', pinned=False, base_tier='high',
        )
