"""What Scheduler.task_priority answers from between ticks (task 9933)."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from orchestrator.overrides import OverrideRow, OverrideSource, OverrideStore
from orchestrator.scheduler_priority_lookup import (
    PAUSED_PIN_REFRESH_SECS,
    LookupPins,
    last_written_tiers,
    pin_orders,
)
from orchestrator.task_priority import PINNED, TaskPriority


def _row(*, pinned: bool, pin_order: int | None = None) -> OverrideRow:
    return OverrideRow(
        boost_tier=None, pinned=pinned, pin_order=pin_order, reserve_now=False,
        ttl_until=None, source=OverrideSource.OPERATOR,
    )


class _CountingStore(OverrideStore):
    reads = 0

    def get_overrides(self, project_root: str) -> dict[str, OverrideRow]:
        self.reads += 1
        return super().get_overrides(project_root)


class _BrokenStore(OverrideStore):
    def get_overrides(self, project_root: str) -> dict[str, OverrideRow]:
        raise OSError('store unreachable')


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == logging.WARNING]


class TestLastWrittenTiers:

    def test_no_path_reads_nothing(self) -> None:
        assert last_written_tiers(None) == {}

    def test_a_missing_file_reads_nothing_silently(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING)

        assert last_written_tiers(tmp_path / 'absent.json') == {}
        assert _warnings(caplog) == []

    @pytest.mark.parametrize('content', [
        b'not json {',
        b'{"effective_priorities": ["not", "a", "map"]}',
    ])
    def test_an_unreadable_snapshot_reads_nothing_and_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, content: bytes,
    ) -> None:
        path = tmp_path / 'scheduler_state.json'
        path.write_bytes(content)
        caplog.set_level(logging.WARNING)

        assert last_written_tiers(path) == {}
        assert len(_warnings(caplog)) == 1


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

    def test_a_failing_store_read_leaves_pins_unknown(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING)

        pins = LookupPins(_BrokenStore(tmp_path / 'overrides.db'), str(tmp_path), clock=lambda: 0.0)

        assert pins.priority('X', 'low').pinned is None
        assert len(_warnings(caplog)) == 1

    @pytest.mark.asyncio
    async def test_a_paused_refresh_reads_at_most_once_a_period(self, tmp_path: Path) -> None:
        now = [0.0]
        store = _CountingStore(tmp_path / 'overrides.db')
        pins = LookupPins(store, str(tmp_path), clock=lambda: now[0])
        store.set_override(str(tmp_path), 'P', pinned=True)

        now[0] = PAUSED_PIN_REFRESH_SECS - 1
        await pins.refresh_if_due()
        assert (store.reads, pins.priority('P', 'low').pinned) == (1, False)

        now[0] = PAUSED_PIN_REFRESH_SECS
        await pins.refresh_if_due()
        assert (store.reads, pins.priority('P', 'low').pinned) == (2, True)
