"""The task picture: what the ordinary pass's read parts see, built without writing anything.

plans/early-start-on-predecessor-branch-prd.md §5.4 "From what: the task
picture", invariant 12, test row 35 (task 10476).
"""

from __future__ import annotations

from collections.abc import MutableMapping

import pytest
from _scheduler_pass_scenario import (
    BOOSTED,
    DONE_EXT_DEP,
    EXPIRED_ROW,
    GREP_FAIL_DEP,
    GREP_PASS_DEP,
    HIGH_DEPENDENT,
    INHERITING_LOW,
    PINNED,
    PLAIN_DONE_DEP,
    PLAIN_DONE_DEPENDENT,
    SCRIPT_DEP,
    SENTINEL_EXT_DEP,
    SYNTHETIC_IDS,
    WATCHDOG_PINNED,
    all_events,
    build_scenario,
    count_script_runs,
    record_passes,
)

from orchestrator.overrides import OverrideRow, OverrideSource
from orchestrator.task_picture import TaskPicture
from orchestrator.task_priority import PINNED as PINNED_TIER
from orchestrator.task_priority import TaskPriority

pytestmark = [pytest.mark.usefixtures('code_default_config'), pytest.mark.asyncio]


async def _picture(scenario) -> TaskPicture:
    picture = await scenario.scheduler.build_task_picture()
    assert isinstance(picture, TaskPicture)
    return picture


class TestContents:
    async def test_dep_statuses_and_records_the_active_fetch_leaves_out(self, tmp_path):
        picture = await _picture(build_scenario(tmp_path))

        assert picture.status_map[GREP_PASS_DEP] == 'done'
        record = picture.terminal_dep_records[GREP_PASS_DEP]
        assert record['id'] == GREP_PASS_DEP
        assert record['metadata']['delivered_checks'][0]['name'] == 'lands'
        assert GREP_PASS_DEP not in picture.tasks_by_id

    async def test_external_statuses_are_the_resolvers_answer(self, tmp_path):
        picture = await _picture(build_scenario(tmp_path))

        assert picture.external_statuses == {SENTINEL_EXT_DEP: 'unknown_task', DONE_EXT_DEP: 'done'}
        assert picture.external_resolver_failed is False

    async def test_delivered_checks_are_evaluated_at_main(self, tmp_path):
        picture = await _picture(build_scenario(tmp_path))

        assert picture.delivered_check_cache == {GREP_PASS_DEP: True, GREP_FAIL_DEP: False, SCRIPT_DEP: True}

    async def test_switched_off_delivered_checks_read_as_none(self, tmp_path):
        picture = await _picture(build_scenario(tmp_path, delivered_checks_enabled=False))

        assert picture.delivered_check_cache is None

    async def test_overrides_are_the_ones_in_force(self, tmp_path):
        picture = await _picture(build_scenario(tmp_path))

        assert set(picture.overrides) == {BOOSTED, PINNED}
        assert {GREP_PASS_DEP, EXPIRED_ROW, WATCHDOG_PINNED}.isdisjoint(picture.overrides)

    async def test_tiers_inherit_and_take_live_boosts_only(self, tmp_path):
        picture = await _picture(build_scenario(tmp_path))

        assert picture.effective_priorities[INHERITING_LOW] == 'high'
        assert picture.effective_priorities[BOOSTED] == 'high'
        assert picture.effective_priorities[EXPIRED_ROW] == 'low'
        assert picture.transitive_counts[INHERITING_LOW] == 1
        assert picture.transitive_counts[HIGH_DEPENDENT] == 0

    async def test_max_id_and_built_at(self, tmp_path):
        picture = await _picture(build_scenario(tmp_path))

        assert picture.max_id == int(PLAIN_DONE_DEPENDENT)
        assert picture.built_at == 1000.0

    async def test_its_tiers_are_the_ordinary_passes_tiers(self, tmp_path):
        picture = await _picture(build_scenario(tmp_path / 'a'))
        twin = build_scenario(tmp_path / 'b')
        await twin.scheduler.acquire_next()

        for task_id in SYNTHETIC_IDS:
            assert twin.scheduler.task_priority(task_id).base_tier == picture.effective_priorities.get(task_id)

    async def test_a_failed_task_fetch_gives_no_picture(self, tmp_path):
        scenario = build_scenario(tmp_path)
        scenario.session.failing.add('get_tasks')

        assert await scenario.scheduler.build_task_picture() is None


class TestWritesNothing:
    """Row 35: building a picture leaves every store, streak and cache as it was."""

    async def test_no_write_event_override_callback_or_tier_changes(self, tmp_path):
        scenario = build_scenario(tmp_path)
        overrides_before = scenario.override_store.get_overrides(scenario.project_root)
        priorities_before = {tid: scenario.scheduler.task_priority(tid) for tid in SYNTHETIC_IDS}

        await _picture(scenario)
        await _picture(scenario)

        assert scenario.session.writes == []
        assert all_events(scenario.event_store) == []
        assert scenario.override_store.get_overrides(scenario.project_root) == overrides_before
        assert scenario.callback_calls == []
        assert {tid: scenario.scheduler.task_priority(tid) for tid in SYNTHETIC_IDS} == priorities_before

    async def test_nothing_it_read_is_cached(self, tmp_path):
        scenario = build_scenario(tmp_path)

        await _picture(scenario)
        first_fetches = scenario.session.count('get_task')
        first_runs = count_script_runs(scenario.script_counter)
        await _picture(scenario)

        assert first_fetches > 0 and first_runs > 0
        assert scenario.session.count('get_task') == 2 * first_fetches
        assert count_script_runs(scenario.script_counter) == 2 * first_runs

    async def test_a_consumers_edit_to_a_picture_reaches_no_later_picture(self, tmp_path):
        scenario = build_scenario(tmp_path)
        await scenario.scheduler.acquire_next()

        first = await _picture(scenario)
        first.terminal_dep_records[PLAIN_DONE_DEP]['status'] = 'pending'
        second = await _picture(scenario)

        assert second.terminal_dep_records[PLAIN_DONE_DEP]['status'] == 'done'

    async def test_its_override_and_tier_maps_are_read_only(self, tmp_path):
        picture = await _picture(build_scenario(tmp_path))

        for mapping in (picture.overrides, picture.effective_priorities, picture.transitive_counts):
            assert not isinstance(mapping, MutableMapping)

    async def test_the_next_ordinary_passes_behave_as_if_no_picture_was_built(self, tmp_path):
        with_picture = build_scenario(tmp_path / 'a')
        without = build_scenario(tmp_path / 'b')

        await _picture(with_picture)
        a = await record_passes(with_picture)
        b = await record_passes(without)

        assert a.decisions == b.decisions
        assert a.get_task_calls == b.get_task_calls
        assert a.script_runs == b.script_runs


def _bare_picture(*, overrides: dict[str, OverrideRow], effective_priorities: dict[str, str]) -> TaskPicture:
    return TaskPicture(
        built_at=0.0, tasks=(), status_map={}, tasks_by_id={}, max_id=0,
        terminal_dep_records={}, deferred_terminal_deps=frozenset(), external_statuses={},
        external_resolver_failed=False, delivered_check_cache=None, overrides=overrides,
        effective_priorities=effective_priorities, transitive_counts={},
    )


class TestPriorityOf:
    """A task's priority in the picture: its base tier combined with the pins in force."""

    async def test_a_pinned_task_reads_pinned_with_its_order_and_base_tier(self, tmp_path):
        picture = await _picture(build_scenario(tmp_path))

        assert picture.priority_of(PINNED) == TaskPriority(
            tier=PINNED_TIER, pinned=True,
            base_tier=picture.effective_priorities[PINNED],
            pin_order=picture.overrides[PINNED].pin_order or 0,
        )

    async def test_a_boosted_task_reads_its_boosted_tier_unpinned(self, tmp_path):
        picture = await _picture(build_scenario(tmp_path))

        assert picture.priority_of(BOOSTED) == TaskPriority(
            tier='high', pinned=False, base_tier='high',
        )

    async def test_a_watchdog_pin_reads_pinned(self):
        watchdog_row = OverrideRow(
            boost_tier=None, pinned=True, pin_order=4, reserve_now=False,
            ttl_until=None, source=OverrideSource.STARVATION_WATCHDOG,
        )
        picture = _bare_picture(overrides={'T': watchdog_row}, effective_priorities={'T': 'low'})

        assert picture.priority_of('T') == TaskPriority(
            tier=PINNED_TIER, pinned=True, base_tier='low', pin_order=4,
        )
